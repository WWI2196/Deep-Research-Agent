"""Lightweight ReAct agent loop for research subagents.

Implements think → act → observe cycles without nesting LangGraph,
keeping debugging simple and preserving existing test stability.

LLM output format (JSON):
  {
    "thought": "reasoning about what to do next",
    "action": "tool_name",
    "action_input": {"arg1": "value1", ...}
  }

Or termination:
  {
    "thought": "I have gathered enough evidence...",
    "final_answer": "# Report Title\n..."
  }
"""

import json
import logging
from collections import defaultdict
from typing import Any, Callable

from .helpers import extract_json
from .tracing import trace

logger = logging.getLogger(__name__)

MAX_STEPS_DEFAULT = 15
MAX_SEARCH_ROUNDS_DEFAULT = 4
MAX_RECENT_OBSERVATIONS = 6
COMPRESSION_THRESHOLD_CHARS = 12000
COMPRESSION_SUMMARY_MAX_LEN = 1500


async def run_react_agent(
    system_prompt: str,
    user_prompt: str,
    tools: list,
    chat_fn: Callable,
    max_steps: int = MAX_STEPS_DEFAULT,
    temperature: float = 0.3,
    role: str = "subagent",
    subtask_id: str | None = None,
    max_search_rounds: int | None = None,
) -> dict[str, Any]:
    """Run a ReAct agent loop.

    Args:
        system_prompt: The system prompt defining the agent's role and available tools.
        user_prompt: The initial user request.
        tools: List of Tool instances from tools.py.
        chat_fn: Async function to call the LLM (typically `chat` from llm.py).
        max_steps: Maximum number of tool-use iterations before forced termination.
        temperature: LLM sampling temperature.
        role: Role label passed to chat_fn (e.g. "subagent", "planner").
        subtask_id: Optional subtask identifier injected into trace events for observability.
        max_search_rounds: Override default search budget. If None, uses MAX_SEARCH_ROUNDS_DEFAULT.

    Returns:
        dict with:
          - final_answer: str
          - tool_calls: list[dict]  # complete history of tool invocations
          - steps_taken: int
    """
    tool_map = {t.name: t for t in tools}
    tool_descriptions = _build_tool_descriptions(tools)

    # Build the initial conversation
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt + "\n\n" + tool_descriptions},
        {"role": "user", "content": user_prompt},
    ]

    tool_calls: list[dict[str, Any]] = []
    synthesize_evidence_used = False
    consecutive_fetch_fails = 0
    search_rounds_used = 0
    _max_search_rounds = max_search_rounds if max_search_rounds is not None else MAX_SEARCH_ROUNDS_DEFAULT
    last_search_result_count: int | None = None
    consecutive_low_results = 0
    seen_source_urls: set[str] = set()
    writing_phase_injected = False

    async def _tr(event_type: str, message: str, details: dict[str, Any] | None = None, level: str = "debug") -> None:
        d = details or {}
        if subtask_id:
            d["subtask_id"] = subtask_id
        await trace("subagents", event_type, message, d, level=level)

    for step in range(max_steps):
        # Early warning: transition to writing when budget is running low
        if step == max_steps - 5:
            messages.append({
                "role": "system",
                "content": (
                    "[SYSTEM] You have 5 steps remaining. Begin transitioning to report writing. "
                    "Finish any ongoing search, then synthesize your findings and prepare your final report."
                ),
            })
            await _tr("react_phase", "Early warning: 5 steps remaining", {"step": step + 1})

        # Writing phase: forcibly disable search/extraction tools so the agent must finalize
        if step >= max_steps - 3:
            removed: list[str] = []
            for name in ("searxng_search", "document_hybrid_search", "evaluate_sources", "fetch_fulltext"):
                if name in tool_map:
                    tool_map.pop(name)
                    removed.append(name)
            if not writing_phase_injected:
                available = ", ".join(tool_map.keys())
                removed_str = f" ({', '.join(removed)} disabled)" if removed else ""
                messages.append({
                    "role": "system",
                    "content": (
                        f"[SYSTEM] WRITING PHASE{removed_str}. "
                        f"Available tools now: {available}. "
                        "You MUST complete and submit your report using existing evidence. "
                        "CRITICAL: When you finish writing, output your FULL report via 'final_answer'. "
                        "The 'final_answer' field must contain the COMPLETE markdown report — "
                        "not a summary, not an outline. Its length should match your submit_report. "
                        "Do NOT call 'final_answer' as a tool; it is a JSON field, not an action."
                    ),
                })
                writing_phase_injected = True
                await _tr("react_phase", f"Writing phase activated{removed_str}", {
                    "step": step + 1,
                    "available_tools": list(tool_map.keys()),
                })

        await _tr("react_step", f"Step {step + 1}/{max_steps}", {
            "step": step + 1,
            "max_steps": max_steps,
        })

        # ── Think ──
        try:
            response = await chat_fn(
                role=role,
                messages=messages,
                temperature=temperature,
            )
        except Exception as exc:
            logger.error("ReAct LLM call failed at step %d: %s", step + 1, exc)
            await _tr("react_error", f"LLM call failed: {exc}", level="error")
            return {
                "final_answer": "",
                "tool_calls": tool_calls,
                "steps_taken": step,
                "error": f"LLM call failed: {exc}",
            }

        content = response.strip()

        # ── Parse ──
        parsed = _parse_react_output(content)
        if not parsed:
            # Try once more with a repair prompt
            repair_msg = (
                "Your previous response could not be parsed. "
                "You MUST respond with valid JSON only.\n\n"
                f"Your previous response:\n{content[:500]}\n\n"
                "Respond with JSON containing either:\n"
                '- {"thought": "...", "action": "tool_name", "action_input": {...}}\n'
                '- {"thought": "...", "final_answer": "..."}'
            )
            messages.append({"role": "user", "content": repair_msg})
            try:
                response = await chat_fn(
                    role="subagent",
                    messages=messages,
                    temperature=temperature,
                )
                parsed = _parse_react_output(response.strip())
            except Exception as exc:
                logger.warning("ReAct repair attempt failed: %s", exc)

        if not parsed:
            logger.warning("ReAct could not parse LLM output after repair at step %d", step + 1)
            messages.append({
                "role": "user",
                "content": (
                    "I could not understand your response. "
                    "Please provide your final answer now if you have enough evidence, "
                    "or use a valid tool call if you need to gather more."
                ),
            })
            continue

        thought = parsed.get("thought", "")

        # ── Termination? ──
        if "final_answer" in parsed:
            final = parsed["final_answer"]
            # Prefer longer submit_report if final_answer is much shorter
            best_report = ""
            for tc in tool_calls:
                if tc.get("tool") == "submit_report":
                    tc_result = tc.get("result", {})
                    if tc_result.get("success"):
                        report_text = tc_result.get("result", {}).get("report", "")
                        if len(report_text) > len(best_report):
                            best_report = report_text
            if best_report and len(final) < len(best_report) * 0.5:
                await _tr("react_recover", "Replaced short final_answer with longer submit_report", {
                    "final_answer_length": len(final),
                    "submit_report_length": len(best_report),
                }, level="info")
                final = best_report
            await _tr("react_complete", f"ReAct complete after {step + 1} steps", {
                "steps": step + 1,
                "report_length": len(final),
            })
            return {
                "final_answer": final,
                "tool_calls": tool_calls,
                "steps_taken": step + 1,
            }

        # ── Act ──
        action = parsed.get("action", "")
        action_input = parsed.get("action_input", {})

        if not action:
            messages.append({
                "role": "user",
                "content": (
                    "You provided a thought but no action. "
                    "Please either call a tool with 'action' and 'action_input', "
                    "or provide 'final_answer' if you are done."
                ),
            })
            continue

        if action not in tool_map:
            available = ", ".join(tool_map.keys())
            messages.append({
                "role": "user",
                "content": (
                    f"Unknown tool '{action}'. Available tools: {available}. "
                    "Please use one of the available tools or provide final_answer."
                ),
            })
            await _tr("react_tool_error", f"Unknown tool: {action}", {
                "available": list(tool_map.keys()),
            }, level="warning")
            continue

        tool = tool_map[action]
        await _tr("react_act", f"Executing {action}", {
            "step": step + 1,
            "tool": action,
            "input_preview": str(action_input)[:200],
        })

        # Guard: synthesize_evidence may only be used once
        if action == "synthesize_evidence":
            if synthesize_evidence_used:
                result = {
                    "success": True,
                    "result": {
                        "message": "synthesize_evidence has already been used. Proceed directly with search or submit_report.",
                        "synthesis": {},
                    },
                    "error": None,
                }
                tool_calls.append({
                    "step": step + 1,
                    "tool": action,
                    "input": action_input,
                    "result": result,
                })
                messages.append({
                    "role": "assistant",
                    "content": json.dumps({"thought": thought, "action": action, "action_input": action_input}, ensure_ascii=False),
                })
                messages.append({
                    "role": "user",
                    "content": f"Observation:\n{result['result']['message']}",
                })
                await _tr("react_guard", "Blocked duplicate synthesize_evidence", {
                    "step": step + 1,
                }, level="warning")
                continue
            synthesize_evidence_used = True

        # Guard: skip fetch_fulltext if previous two attempts yielded nothing
        if action == "fetch_fulltext" and consecutive_fetch_fails >= 2:
            result = {
                "success": True,
                "result": {
                    "extracted": {},
                    "failed_count": 0,
                    "message": "Skipped: previous fetch attempts returned no content. Use available search snippets instead.",
                },
                "error": None,
            }
            tool_calls.append({
                "step": step + 1,
                "tool": action,
                "input": action_input,
                "result": result,
            })
            messages.append({
                "role": "assistant",
                "content": json.dumps({"thought": thought, "action": action, "action_input": action_input}, ensure_ascii=False),
            })
            messages.append({
                "role": "user",
                "content": f"Observation:\n{result['result']['message']}",
            })
            await _tr("react_guard", "Blocked fetch_fulltext after 2 consecutive failures", {
                "step": step + 1,
            }, level="warning")
            continue

        # Guard: limit search rounds dynamically
        if action in ("searxng_search", "document_hybrid_search"):
            if search_rounds_used >= _max_search_rounds:
                result = {
                    "success": True,
                    "result": {
                        "query": action_input.get("query", ""),
                        "results": [],
                        "source": action,
                        "message": (
                            f"[SYSTEM] Search round limit reached ({search_rounds_used}/{_max_search_rounds}). "
                            "NO MORE SEARCHES ARE ALLOWED. "
                            "You MUST write your report NOW using only the evidence you have already gathered. "
                            "Any further attempt to call a search tool will be REJECTED. "
                            "Call submit_report immediately with your final markdown report."
                        ),
                    },
                    "error": None,
                }
                tool_calls.append({
                    "step": step + 1,
                    "tool": action,
                    "input": action_input,
                    "result": result,
                })
                messages.append({
                    "role": "assistant",
                    "content": json.dumps({"thought": thought, "action": action, "action_input": action_input}, ensure_ascii=False),
                })
                messages.append({
                    "role": "user",
                    "content": f"Observation:\n{result['result']['message']}",
                })
                await _tr("react_guard", f"Blocked search: max {_max_search_rounds} rounds reached", {
                    "step": step + 1,
                    "max_search_rounds": _max_search_rounds,
                }, level="warning")
                # Remove search tools from available set so LLM cannot call them again
                for name in ("searxng_search", "document_hybrid_search"):
                    tool_map.pop(name, None)
                continue
            search_rounds_used += 1

        try:
            result = await tool.execute(**action_input)
        except Exception as exc:
            result = {"success": False, "result": None, "error": str(exc)}
            logger.warning("Tool %s execution error: %s", action, exc)

        # Track search result counts for dynamic round adjustment
        if action in ("searxng_search", "document_hybrid_search") and result.get("success"):
            results_list = result.get("result", {}).get("results", [])
            result_count = len(results_list) if isinstance(results_list, list) else 0
            limit = action_input.get("limit", 8)
            if result_count < limit * 0.5:
                consecutive_low_results += 1
            else:
                consecutive_low_results = 0
            if consecutive_low_results >= 1:
                _max_search_rounds = 5
            last_search_result_count = result_count

            # Track source duplication within this subagent
            dupes = [r for r in results_list if r.get("url") in seen_source_urls]
            if results_list and len(dupes) >= len(results_list) * 0.5:
                result["result"]["duplicate_note"] = (
                    "[SYSTEM NOTE] Over 50% of these results are sources you have already seen. "
                    "You are repeating searches. STOP searching and write your report using existing evidence."
                )
            for r in results_list:
                if url := r.get("url"):
                    seen_source_urls.add(url)

        # Track consecutive fetch_fulltext failures
        if action == "fetch_fulltext":
            extracted = result.get("result", {}).get("extracted", {}) if result.get("success") else {}
            if not extracted:
                consecutive_fetch_fails += 1
            else:
                consecutive_fetch_fails = 0

        tool_calls.append({
            "step": step + 1,
            "tool": action,
            "input": action_input,
            "result": result,
        })

        # Guard: prevent premature submit_report without evidence
        if action == "submit_report":
            search_calls = [
                tc for tc in tool_calls
                if tc.get("tool") in ("searxng_search", "document_hybrid_search")
            ]
            if len(search_calls) < 1:
                messages.append({
                    "role": "assistant",
                    "content": json.dumps({"thought": thought, "action": action, "action_input": action_input}, ensure_ascii=False),
                })
                messages.append({
                    "role": "user",
                    "content": (
                        "Observation: You tried to submit a report but have not gathered any evidence yet. "
                        "Please use searxng_search or document_hybrid_search to find sources first. "
                        "If you are unsure what to search for, use synthesize_evidence to plan your approach."
                    ),
                })
                await _tr("react_guard", "Blocked premature submit_report", {
                    "step": step + 1,
                    "search_calls": 0,
                }, level="warning")
                continue
            # Fix any tool-name URLs in evidence before submitting
            action_input = _fix_submit_report_urls(action_input, tool_calls)

            # Guard: reject reports that are too short
            if result.get("success"):
                report_text = result.get("result", {}).get("report", "")
                if len(report_text) < 800:
                    messages.append({
                        "role": "assistant",
                        "content": json.dumps({"thought": thought, "action": action, "action_input": action_input}, ensure_ascii=False),
                    })
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Observation: Report submitted but is TOO SHORT ({len(report_text)} characters). "
                            "A complete subagent report should be 800-1500 words with detailed analysis and evidence assessment. "
                            "Please expand your report with more depth, examples, and source citations before submitting again."
                        ),
                    })
                    await _tr("react_guard", f"Rejected short report ({len(report_text)} chars)", {
                        "step": step + 1,
                    }, level="warning")
                    continue

        # ── Observe ──
        observation = _format_observation(result)
        if action in ("searxng_search", "document_hybrid_search") and consecutive_low_results >= 2:
            observation += (
                "\n\n[SYSTEM NOTE] Previous searches returned very few results. "
                "Try broader or alternative queries, or use different source types."
            )
        messages.append({
            "role": "assistant",
            "content": json.dumps({"thought": thought, "action": action, "action_input": action_input}, ensure_ascii=False),
        })
        messages.append({
            "role": "user",
            "content": f"Observation:\n{observation}",
        })

        await _tr("react_observe", f"Observation from {action}", {
            "step": step + 1,
            "tool": action,
            "success": result.get("success"),
            "observation_preview": observation[:200],
        })

        # ── Context compression ──
        if _should_compress_messages(messages):
            messages = await _compress_messages(
                messages, chat_fn, temperature,
                keep_recent=MAX_RECENT_OBSERVATIONS,
                trace_fn=_tr,
            )

    # ── Max steps reached ──
    logger.warning("ReAct reached max_steps (%d) without final_answer", max_steps)
    await _tr("react_max_steps", f"Max steps reached ({max_steps})", level="warning")

    # Find the best report previously generated by submit_report
    best_report = ""
    for tc in tool_calls:
        if tc.get("tool") == "submit_report":
            tc_result = tc.get("result", {})
            if tc_result.get("success"):
                report_text = tc_result.get("result", {}).get("report", "")
                if len(report_text) > len(best_report):
                    best_report = report_text

    # One final attempt to get an answer
    messages.append({
        "role": "user",
        "content": (
            "[SYSTEM] You have reached the MAXIMUM number of steps. "
            "This is your FINAL chance to respond. "
            "You MUST provide your final_answer NOW. "
            "Do NOT call any tools. Write your report based on all evidence gathered so far. "
            "IMPORTANT: 'final_answer' is a JSON FIELD containing your COMPLETE markdown report, "
            "NOT a tool name. Output: {\"thought\": \"...\", \"final_answer\": \"# Report Title\\n...\"}"
        ),
    })
    try:
        final_response = await chat_fn(
            role="subagent",
            messages=messages,
            temperature=temperature,
        )
        final_parsed = _parse_react_output(final_response.strip())
        if final_parsed and "final_answer" in final_parsed:
            final_answer = final_parsed["final_answer"]
            # Prefer the longer submit_report if the final answer is much shorter
            if best_report and len(final_answer) < len(best_report) * 0.5:
                logger.info(
                    "Using longer submit_report (%d chars) instead of short final_answer (%d chars)",
                    len(best_report), len(final_answer),
                )
                await _tr("react_recover", "Recovered longer submit_report over short final_answer", {
                    "submit_report_length": len(best_report),
                    "final_answer_length": len(final_answer),
                }, level="info")
                final_answer = best_report
            return {
                "final_answer": final_answer,
                "tool_calls": tool_calls,
                "steps_taken": max_steps,
            }
    except Exception as exc:
        logger.error("Final answer attempt after max_steps failed: %s", exc)

    # If final attempt failed but we have a previous submit_report, use it
    if best_report:
        await _tr("react_recover", "Used previous submit_report after failed final attempt", {
            "report_length": len(best_report),
        }, level="info")
        return {
            "final_answer": best_report,
            "tool_calls": tool_calls,
            "steps_taken": max_steps,
        }

    return {
        "final_answer": "",
        "tool_calls": tool_calls,
        "steps_taken": max_steps,
        "error": f"Reached max_steps ({max_steps}) without producing final_answer",
    }


def _build_tool_descriptions(tools: list) -> str:
    """Build a prompt section describing available tools."""
    lines = ["You have access to the following tools:"]
    for t in tools:
        params = ", ".join(f"{k}: {v}" for k, v in t.params_schema.items())
        lines.append(f"- {t.name}: {t.description} (params: {params})")
    lines.append(
        "\nCRITICAL RULES:\n"
        "- You have a LIMITED search budget. After search budget is exhausted, "
        "you MUST call submit_report immediately. Do NOT attempt additional searches.\n"
        "- Respond with JSON only. Either:\n"
        '  1. {"thought": "...", "action": "tool_name", "action_input": {"arg": "value"}}\n'
        '  2. {"thought": "...", "final_answer": "your complete markdown report here"}\n'
        "- 'final_answer' is a JSON FIELD containing your COMPLETE markdown report, NOT a tool to call. "
        "It must be the FULL report text (800-1500+ words), not a summary or outline.\n"
        "- Do not wrap JSON in markdown code blocks."
    )
    return "\n".join(lines)


def _parse_react_output(text: str) -> dict[str, Any] | None:
    """Parse LLM output into ReAct structure. Handles raw JSON and markdown-wrapped JSON."""
    text = text.strip()
    # Remove markdown code fences if present
    if text.startswith("```"):
        text = text[text.find("\n") + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()
    if text.startswith("json"):
        text = text[text.find("\n") + 1:].strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try extract_json fallback
    extracted = extract_json(text)
    if extracted:
        try:
            parsed = json.loads(extracted)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def _format_observation(result: dict[str, Any]) -> str:
    """Format tool result into a concise observation string for the LLM."""
    if not result.get("success"):
        error = result.get("error", "Unknown error")
        return f"Tool execution failed: {error}"

    data = result.get("result", {})

    # Truncate large results to avoid context explosion
    MAX_OBSERVATION_LEN = 4000

    # Custom formatting for search results
    if isinstance(data, dict) and "results" in data:
        results = data["results"]
        lines = [f"Found {len(results)} results:"]
        for i, r in enumerate(results[:8], 1):
            title = r.get("title", "")
            url = r.get("url", "")
            desc = (r.get("description", "") or r.get("snippet", ""))[:200]
            score = r.get("quality_score") or r.get("score", "")
            score_str = f" (score: {score:.2f})" if isinstance(score, float) else ""
            lines.append(f"{i}. {title}{score_str}\n   URL: {url}\n   {desc}")
        if len(results) > 8:
            lines.append(f"... and {len(results) - 8} more results")
        obs = "\n\n".join(lines)
        if data.get("duplicate_note"):
            obs += "\n\n" + data["duplicate_note"]
    elif isinstance(data, dict) and "extracted" in data:
        extracted = data["extracted"]
        lines = [f"Extracted full text from {len(extracted)} URLs:"]
        for url, text in list(extracted.items())[:5]:
            preview = text[:500].replace("\n", " ") if text else "(failed)"
            lines.append(f"- {url}:\n  {preview}...")
        obs = "\n\n".join(lines)
    elif isinstance(data, dict) and "scored" in data:
        scored = data["scored"]
        selected = data.get("selected_for_fulltext", [])
        lines = [
            f"Evaluated {len(scored)} sources.",
            f"Selected {len(selected)} for full-text extraction: {', '.join(selected[:5])}",
        ]
        for i, s in enumerate(scored[:5], 1):
            score = s.get("quality_score", 0)
            ft = "FULLTEXT" if s.get("full_text") else "snippet"
            lines.append(f"{i}. [{ft}] {s.get('title', '')} (score: {score:.2f}) — {s.get('url', '')}")
        obs = "\n".join(lines)
    elif isinstance(data, dict) and "report" in data:
        report = data["report"]
        obs = f"Report generated ({len(report)} characters).\n\n{report[:1000]}"
    else:
        obs = json.dumps(data, ensure_ascii=False, default=str)

    if len(obs) > MAX_OBSERVATION_LEN:
        obs = obs[:MAX_OBSERVATION_LEN] + "\n\n[Observation truncated due to length]"

    return obs


def _should_compress_messages(messages: list[dict[str, str]]) -> bool:
    """Check if the accumulated message content exceeds the compression threshold."""
    total = sum(len(m.get("content", "")) for m in messages[2:])
    return total > COMPRESSION_THRESHOLD_CHARS


async def _compress_messages(
    messages: list[dict[str, str]],
    chat_fn: Callable,
    temperature: float,
    keep_recent: int = MAX_RECENT_OBSERVATIONS,
    trace_fn: Callable | None = None,
) -> list[dict[str, str]]:
    """Compress older messages, keeping the most recent N observation rounds intact.

    Strategy:
      - Preserve system prompt + initial user prompt (first 2 messages).
      - Preserve the last `keep_recent` full tool-call rounds (assistant action + user observation).
      - Summarise everything in between via a dedicated LLM call.
      - On compression failure, fall back to a simple truncation notice.
    """
    if len(messages) <= 2 + keep_recent * 2:
        return messages

    head = messages[:2]
    tail = messages[-keep_recent * 2:]
    middle = messages[2:-keep_recent * 2] if keep_recent > 0 else messages[2:]

    history_text = ""
    for m in middle:
        role = m.get("role", "unknown")
        content = m.get("content", "")[:1500]
        history_text += f"[{role.upper()}]\n{content}\n\n"

    compression_prompt = (
        "Summarize the following research tool-call history into a compact paragraph "
        f"(max {COMPRESSION_SUMMARY_MAX_LEN} characters). Preserve key factual findings, "
        "important entities/names/dates/numbers, search outcomes, and any failed attempts.\n\n"
        f"{history_text}\nSummary:"
    )

    if trace_fn is None:
        async def _default_trace(event_type: str, message: str, details: dict[str, Any] | None = None, level: str = "debug") -> None:
            await trace("subagents", event_type, message, details, level=level)
        _trace = _default_trace
    else:
        _trace = trace_fn
    try:
        summary = await chat_fn(
            role="subagent",
            messages=[{"role": "user", "content": compression_prompt}],
            temperature=temperature,
        )
        summary = summary.strip()
        if len(summary) > COMPRESSION_SUMMARY_MAX_LEN:
            summary = summary[:COMPRESSION_SUMMARY_MAX_LEN] + "..."
    except Exception as exc:
        logger.warning("Message compression failed: %s", exc)
        await _trace("react_compress_error", f"Compression failed: {exc}", level="warning")
        summary = (
            "[Previous tool calls were truncated due to length. "
            f"Only the most recent {keep_recent} tool-call rounds are shown in full.]"
        )

    await _trace("react_compress", "Context compressed", {
        "original_messages": len(messages),
        "kept_recent": keep_recent,
        "compressed_messages": len(middle),
        "summary_length": len(summary),
    }, level="debug")

    compressed_msg = {
        "role": "user",
        "content": (
            "Previously, you performed several tool calls. "
            f"Here is a summary of what happened:\n\n{summary}"
        ),
    }

    return [*head, compressed_msg, *tail]


def _fix_submit_report_urls(
    action_input: dict[str, Any],
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """Replace tool-name placeholders in submit_report evidence with real URLs.

    LLMs sometimes mistakenly use the tool name (e.g. 'document_hybrid_search')
    as the URL in evidence entries. This function maps those back to the actual
    result URLs found in the tool call history.
    """
    evidence = action_input.get("evidence")
    if not evidence or not isinstance(evidence, list):
        return action_input

    # Build a lookup of tool_name -> list of result URLs from prior tool calls
    url_pool: dict[str, list[str]] = defaultdict(list)
    for tc in tool_calls:
        tool_name = tc.get("tool", "")
        result = tc.get("result", {})
        if not result.get("success"):
            continue
        data = result.get("result", {})
        for item in data.get("results", []):
            url = item.get("url")
            if url:
                url_pool[tool_name].append(url)

    fixed_evidence: list[dict[str, Any]] = []
    for idx, ev in enumerate(evidence):
        if not isinstance(ev, dict):
            fixed_evidence.append(ev)
            continue
        url = ev.get("url", "")
        # Detect tool-name placeholders
        if url in ("searxng_search", "document_hybrid_search"):
            candidates = url_pool.get(url, [])
            if candidates:
                # Assign round-robin if multiple results exist
                real_url = candidates[idx % len(candidates)]
                ev = {**ev, "url": real_url}
        fixed_evidence.append(ev)

    return {**action_input, "evidence": fixed_evidence}

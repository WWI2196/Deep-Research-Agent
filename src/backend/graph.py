"""LangGraph state graph for the research pipeline — all async."""

import asyncio
import json
import logging
import uuid
from typing import Any

from langgraph.graph import END, StateGraph

from .config import get_config
from .helpers import extract_json, needs_continuation
from .llm import chat
from .models import ResearchState
from .persistence import (
    persist_checkpoint,
    persist_run,
    persist_source,
    persist_subagent_report,
    update_run_status,  # noqa: F401 — used by server.py tests
)
from .planning import generate_research_plan, split_into_subtasks
from .prompts import REFLECTION
from .subagent import run_subagent, run_subagents_parallel
from .synthesis import add_citations, synthesize_report
from .tracing import current_phase, current_run_id, trace

logger = logging.getLogger(__name__)

PHASE_WEIGHTS = {
    "init": 2, "plan": 8, "split": 5,
    "subagents": 58, "reflection": 5, "synthesize": 12, "cite": 8,
}
TOTAL_WEIGHT = sum(PHASE_WEIGHTS.values())


async def build_and_run_graph(state: dict[str, Any], on_event, cancel_event=None) -> dict[str, Any]:
    """Build the LangGraph state machine and run it, emitting events along the way."""

    current_run_id.set(state.get("run_id", ""))
    cfg = get_config()
    completed_weight = 0

    async def _check_cancelled() -> None:
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError(f"Research {state.get('run_id', '')} cancelled by user")

    async def _emit(evt: dict[str, Any]) -> None:
        evt.setdefault("run_id", state.get("run_id", ""))
        await on_event(evt)

    async def _init_node(s: ResearchState) -> ResearchState:
        current_phase.set("init")
        await _check_cancelled()
        nonlocal completed_weight
        s["run_id"] = s.get("run_id") or str(uuid.uuid4())
        current_run_id.set(s["run_id"])
        s["events"] = []
        s["errors"] = []
        s["subagent_reports"] = []
        s["sources"] = []
        s["completed_subtasks"] = []
        s["iteration_count"] = 0
        s["max_iterations"] = s.get("max_iterations") or 2
        s["research_complete"] = False
        s["quality_threshold"] = s.get("quality_threshold") or cfg.quality_threshold
        s["current_quality_score"] = 0.0
        s["memory"] = {}
        s["synthesis_retry_count"] = 0
        s["synthesis_failure_summary"] = ""
        s["context_compress_retries"] = (
            s.get("context_compress_retries") or cfg.context_compress_retries
        )
        s["keep_tool_results"] = s.get("keep_tool_results") or cfg.keep_tool_results
        s["query_cache"] = {}
        s["document_collections"] = s.get("document_collections") or []
        s["gap_instructions"] = []
        s["tool_call_history"] = []
        s["_subtask_report_map"] = {}

        await trace("init", "node_enter", "Entering init node", {"run_id": s["run_id"], "query": s["user_query"]})
        await _emit({"type": "phase-update", "phase": "init",
               "message": f"Initialised (model: {cfg.default_model})"})
        await _emit({"type": "progress", "phase": "init", "percent": 0})
        completed_weight += PHASE_WEIGHTS["init"]
        await trace("init", "node_exit", "Init complete", {"model": cfg.default_model})

        await persist_run(s["run_id"], s["user_query"], cfg.base_url, cfg.default_model)
        await persist_checkpoint(s["run_id"], "init", s)
        return s

    async def _plan_node(s: ResearchState) -> ResearchState:
        current_phase.set("plan")
        nonlocal completed_weight
        await _check_cancelled()
        await trace("plan", "node_enter", "Entering plan node")
        await _emit({"type": "phase-update", "phase": "plan", "message": "Generating research plan..."})
        plan = await generate_research_plan(
            s["user_query"],
            document_collections=s.get("document_collections") or None,
        )
        s["research_plan"] = plan
        s["plan_methodology"] = plan.get("methodology", "")
        dims = plan.get("dimensions", [])
        preview = json.dumps([d.get("name", "") for d in dims], ensure_ascii=False)
        await _emit({"type": "plan-generated",
               "plan_preview": preview,
               "plan_length": len(json.dumps(plan, ensure_ascii=False)),
               "dimensions": len(dims)})
        await trace("plan", "node_exit", "Plan generated", {"dimensions": len(dims), "plan_preview": preview[:200]})

        pct = _progress(completed_weight, PHASE_WEIGHTS["plan"])
        await _emit({"type": "progress", "phase": "plan", "percent": pct})
        completed_weight += PHASE_WEIGHTS["plan"]

        await persist_checkpoint(s["run_id"], "plan", s)
        return s

    async def _split_node(s: ResearchState) -> ResearchState:
        current_phase.set("split")
        nonlocal completed_weight
        await _check_cancelled()
        await trace("split", "node_enter", "Entering split node")
        await _emit({"type": "phase-update", "phase": "split", "message": "Creating subtasks..."})
        try:
            s["subtasks"] = await split_into_subtasks(s["research_plan"])
            await trace("split", "node_exit", "Subtasks created", {"count": len(s["subtasks"])})
        except Exception as e:
            logger.warning("Split failed, fallback: %s", e)
            await trace("split", "error", f"Split failed: {e}", level="warning")
            plan = s.get("research_plan", {})
            plan_text = json.dumps(plan, ensure_ascii=False)
            s["subtasks"] = [{
                "id": "main", "title": s["user_query"][:80],
                "description": plan_text[:500],
                "objective": s["user_query"],
                "output_format": "markdown",
                "dimension": "main",
                "keywords": plan.get("dimensions", [{}])[0].get("keywords", []) if plan.get("dimensions") else [],
                "source_types": "academic, official, news",
                "boundaries": "",
                "estimated_searches": 10,
            }]
            await trace("split", "node_exit", "Subtasks fallback", {"count": len(s["subtasks"])})

        await _emit({"type": "subtasks-created", "count": len(s["subtasks"]),
               "subtasks": [{"id": t["id"], "title": t["title"],
                             "description": t.get("description", "")[:150],
                             "objective": t.get("objective", "")[:100]}
                            for t in s["subtasks"]]})

        pct = _progress(completed_weight, PHASE_WEIGHTS["split"])
        await _emit({"type": "progress", "phase": "split", "percent": pct})
        completed_weight += PHASE_WEIGHTS["split"]

        await persist_checkpoint(s["run_id"], "split", s)
        return s

    async def _subagents_node(s: ResearchState) -> ResearchState:
        current_phase.set("subagents")
        nonlocal completed_weight
        await _check_cancelled()
        iteration = s.get("iteration_count", 0)
        completed = set(s.get("completed_subtasks", []))
        to_run = [t for t in s.get("subtasks", []) if t.get("id", "") not in completed]
        if not to_run:
            await trace("subagents", "node_exit", "No subagents to run")
            return s

        # Use subtask-level estimated_searches as budget, default 10
        budget = max(
            t.get("estimated_searches", 10) for t in to_run
        ) if to_run else 10

        await trace("subagents", "node_enter", f"Running {len(to_run)} subagents (iteration {iteration + 1})", {"budget": budget, "document_collections": s.get("document_collections", [])})
        await _emit({"type": "phase-update", "phase": "subagents",
               "message": f"Running {len(to_run)} subagents (iteration {iteration + 1})..."})
        await _emit({"type": "subagents-launch", "iteration": iteration + 1, "total_agents": len(to_run),
               "agent_details": [{"id": t["id"], "title": t["title"],
                                  "description": t.get("description", "")[:200]}
                                 for t in to_run]})

        results = await run_subagents_parallel(
            s["user_query"], json.dumps(s.get("research_plan", {}), ensure_ascii=False),
            to_run, budget,
            query_cache=s.get("query_cache", {}),
            document_collections=s.get("document_collections", []),
            gap_instructions=s.get("gap_instructions", []),
        )
        await _check_cancelled()

        # Consume gap_instructions for completed subtasks
        completed_ids = {r["subtask_id"] for r in results.get("raw", [])}
        s["gap_instructions"] = [
            g for g in s.get("gap_instructions", [])
            if g.get("target_subtask_id") not in completed_ids
        ]

        # Accumulate tool_call_history from ReAct subagents
        for r in results.get("raw", []):
            if "tool_calls" in r:
                s.setdefault("tool_call_history", []).extend(r["tool_calls"])

        existing_sources = s.get("sources", [])
        # When reflection re-runs a subtask, replace its old report rather than
        # appending a duplicate. Use subtask_id as the precise key.
        new_raw = results.get("raw", [])
        report_map = s.get("_subtask_report_map", {})
        for r in new_raw:
            report_map[r["subtask_id"]] = r["report"]
        s["_subtask_report_map"] = report_map
        s["subagent_reports"] = list(report_map.values())
        new_completed = [r["subtask_id"] for r in new_raw]
        s["completed_subtasks"] = list(completed | set(new_completed))

        all_sources = existing_sources + results["sources"]
        seen_urls: set[str] = set()
        seen_doc_ids: set[str] = set()
        unique: list[dict] = []
        for src in sorted(all_sources, key=lambda x: x.get("quality_score", 0), reverse=True):
            u = src.get("url")
            if not u:
                continue
            # For document library sources, deduplicate by doc_id
            if u.startswith("file://") or u.startswith("doc://"):
                doc_id = src.get("doc_id", "")
                if not doc_id:
                    # Fallback: extract doc_id from URL (filename without extension)
                    from pathlib import Path
                    doc_id = Path(u).stem
                if doc_id and doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    seen_urls.add(u)
                    unique.append(src)
            else:
                # Web sources: deduplicate by URL as before
                if u not in seen_urls:
                    seen_urls.add(u)
                    unique.append(src)
        s["sources"] = unique
        s["iteration_count"] = iteration + 1

        # Emit per-subagent events & persist
        for item in results.get("raw", []):
            await _emit({"type": "subagent-complete",
                   "subtask_id": item["subtask_id"],
                   "subtask_title": item.get("subtask_title", ""),
                   "report_length": len(item.get("report", "")),
                   "sources_count": len(item.get("sources", [])),
                   "evidence_count": item.get("evidence_count", 0)})

            await persist_subagent_report(
                s["run_id"], item["subtask_id"], item.get("report", ""),
                sources_count=len(item.get("sources", [])),
                evidence_count=item.get("evidence_count", 0),
            )

            for src in item.get("sources", []):
                await persist_source(
                    s["run_id"], src.get("url", ""),
                    title=src.get("title", ""),
                    quality_score=src.get("quality_score", 0),
                    domain=src.get("domain", ""),
                    subtask_id=item["subtask_id"],
                )

        await trace("subagents", "node_exit", f"Subagents complete", {"success": results.get("success_count", 0), "total": results.get("total_count", 0), "new_sources": len(results.get("sources", []))})
        # Split subagents weight across possible iterations to avoid jumping to 99%
        max_iter = max(s.get("max_iterations", 2), 1)
        subagent_weight = PHASE_WEIGHTS["subagents"] / max_iter
        pct = _progress(completed_weight, subagent_weight)
        await _emit({"type": "progress", "phase": "subagents", "percent": pct})
        completed_weight += subagent_weight

        await update_run_status(
            s["run_id"], "running",
            total_sources=len(s.get("sources", [])),
            total_reports=len(s.get("subagent_reports", [])),
            iterations=s.get("iteration_count", 0),
        )
        await persist_checkpoint(s["run_id"], f"subagents_{iteration}", s)
        return s

    async def _reflection_node(s: ResearchState) -> ResearchState:
        current_phase.set("reflection")
        nonlocal completed_weight
        await _check_cancelled()
        iteration = s.get("iteration_count", 0)
        max_iter = s.get("max_iterations", 2)
        reports = s.get("subagent_reports", [])
        subtasks = s.get("subtasks", [])
        past = ", ".join(f"{t.get('id')}: {t.get('title')}" for t in subtasks)

        if iteration >= max_iter:
            quality = s.get("current_quality_score", 0.0)
            threshold = s.get("quality_threshold", 0.6)
            retry_count = s.get("synthesis_retry_count", 0)
            max_retries = s.get("context_compress_retries", cfg.context_compress_retries)

            if quality < threshold and retry_count < max_retries:
                from .synthesis import _generate_failure_summary
                summary = await _generate_failure_summary(
                    user_query=s["user_query"],
                    research_plan=json.dumps(s.get("research_plan", {}), ensure_ascii=False),
                    reports=s.get("subagent_reports", []),
                    partial_report="",
                    reason=f"low-quality ({quality:.2f} below threshold {threshold})",
                    chat_fn=chat,
                )
                s["synthesis_failure_summary"] = summary
                s["synthesis_retry_count"] = retry_count + 1
                await _emit({"type": "reflection-decision", "decision": "low-quality-retry",
                       "quality_score": quality, "threshold": threshold,
                       "iteration": iteration, "retry_count": retry_count + 1})
                await trace("reflection", "decision", "Low quality retry", {"quality": quality, "threshold": threshold, "retry_count": retry_count + 1})
            else:
                await _emit({"type": "reflection-decision", "decision": "max-iterations-reached",
                       "iteration": iteration})
                await trace("reflection", "decision", "Max iterations reached", {"iteration": iteration})

            s["research_complete"] = True
            await trace("reflection", "node_exit", "Reflection complete (max iterations)")
            return s

        await trace("reflection", "node_enter", f"Reflecting (iter {iteration}/{max_iter})")
        await _emit({"type": "phase-update", "phase": "reflection",
               "message": f"Reflecting (iter {iteration}/{max_iter})..."})

        try:
            plan_dims = json.dumps(
                s.get("research_plan", {}).get("dimensions", []),
                ensure_ascii=False
            )
            truncated = "\n\n".join(r[:3000] for r in reports)
            response = await chat(
                role="reflection",
                messages=[{
                    "role": "user",
                    "content": REFLECTION.format(
                        user_query=s["user_query"],
                        research_plan=plan_dims,
                        methodology=s.get("plan_methodology", ""),
                        past_subtasks=past,
                        subagent_reports=truncated,
                    ) + "\n\nReturn ONLY valid JSON.",
                }],
            )
            content = response.strip()
            new_subtasks = []
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                ext = extract_json(content)
                if ext:
                    try:
                        payload = json.loads(ext)
                    except json.JSONDecodeError:
                        payload = {}
                else:
                    payload = {}

            # Extract gaps (limit 3) — distinguish new_subtask vs supplement_existing
            raw_gaps = payload.get("gaps", [])[:3]
            new_subtasks: list[dict[str, Any]] = []
            new_gap_instructions: list[dict[str, Any]] = []
            re_run_subtask_ids: set[str] = set()

            for gap in raw_gaps:
                gap_type = gap.get("gap_type", "new_subtask")
                st = gap.get("subtask", {})

                if gap_type == "supplement_existing":
                    target_id = gap.get("target_subtask_id", "")
                    # Validate target exists among completed subtasks
                    if target_id and target_id in s.get("completed_subtasks", []):
                        new_gap_instructions.append({
                            "target_subtask_id": target_id,
                            "gap_type": gap.get("gap_detail", "missing_evidence"),
                            "description": gap.get("gap_detail", ""),
                            "suggested_queries": gap.get("suggested_queries", []),
                        })
                        re_run_subtask_ids.add(target_id)
                    elif st.get("title"):
                        # Fallback: treat as new subtask if target not found
                        if not st.get("id"):
                            st["id"] = f"gap-fallback-{len(new_subtasks)}"
                        new_subtasks.append(st)
                else:
                    # new_subtask (default)
                    if st.get("title"):
                        if not st.get("id"):
                            st["id"] = f"gap-{len(new_subtasks)}"
                        new_subtasks.append(st)

            overall = payload.get("overall_score", 0)
            previous_score = s.get("current_quality_score", 0.0)
            s["current_quality_score"] = overall

            # Minimum improvement gate: if score gain is too small, skip gap creation
            if overall - previous_score < 0.08:
                new_subtasks.clear()
                new_gap_instructions.clear()
                re_run_subtask_ids.clear()

            # Limit supplement_existing to once per target_subtask_id
            supplemented = set(s.get("supplemented_subtasks", []))
            filtered_gap_instructions: list[dict[str, Any]] = []
            filtered_re_run: set[str] = set()
            for gi in new_gap_instructions:
                tid = gi.get("target_subtask_id", "")
                if tid and tid not in supplemented:
                    filtered_gap_instructions.append(gi)
                    filtered_re_run.add(tid)
                    supplemented.add(tid)
            new_gap_instructions = filtered_gap_instructions
            re_run_subtask_ids = filtered_re_run
            s["supplemented_subtasks"] = list(supplemented)

            # Remove re-run subtasks from completed so subagents_node will re-execute them
            if re_run_subtask_ids:
                s["completed_subtasks"] = [
                    cid for cid in s.get("completed_subtasks", [])
                    if cid not in re_run_subtask_ids
                ]
                s.setdefault("gap_instructions", []).extend(new_gap_instructions)

            if new_subtasks or new_gap_instructions:
                await _emit({"type": "reflection-decision", "decision": "gaps-found",
                       "new_subtask_count": len(new_subtasks),
                       "gap_instruction_count": len(new_gap_instructions),
                       "new_subtasks": [{"id": t.get("id", ""), "title": t.get("title", "")}
                                        for t in new_subtasks],
                       "overall_score": overall,
                       "iteration": iteration})
                await trace("reflection", "decision", "Gaps found", {
                    "new_subtasks": len(new_subtasks),
                    "gap_instructions": len(new_gap_instructions),
                    "overall_score": overall,
                })
                s["subtasks"].extend(new_subtasks)
                s["research_complete"] = False
            else:
                await _emit({"type": "reflection-decision", "decision": "research-complete",
                       "iteration": iteration,
                       "total_reports": len(reports),
                       "total_sources": len(s.get("sources", [])),
                       "overall_score": overall})
                await trace("reflection", "decision", "Research complete", {"overall_score": overall, "total_reports": len(reports)})
                s["research_complete"] = True
        except Exception as e:
            logger.warning("Reflection failed: %s", e)
            await trace("reflection", "error", f"Reflection failed: {e}", level="warning")
            s["research_complete"] = True

        pct = _progress(completed_weight, PHASE_WEIGHTS["reflection"])
        await _emit({"type": "progress", "phase": "reflection", "percent": pct})
        completed_weight += PHASE_WEIGHTS["reflection"]
        await trace("reflection", "node_exit", "Reflection complete")

        await update_run_status(
            s["run_id"], "running",
            total_sources=len(s.get("sources", [])),
            total_reports=len(s.get("subagent_reports", [])),
            iterations=s.get("iteration_count", 0),
        )
        await persist_checkpoint(s["run_id"], "reflection", s)
        return s

    async def _synthesize_node(s: ResearchState) -> ResearchState:
        current_phase.set("synthesize")
        nonlocal completed_weight
        await _check_cancelled()
        report_count = len(s.get("subagent_reports", []))
        await trace("synthesize", "node_enter", f"Synthesizing {report_count} reports", {"report_count": report_count, "retry_count": s.get("synthesis_retry_count", 0)})
        await _emit({"type": "phase-update", "phase": "synthesize",
               "message": f"Synthesizing {report_count} reports..."})

        plan = s.get("research_plan", {})
        plan_text = json.dumps(plan, ensure_ascii=False)
        failure_summary = s.get("synthesis_failure_summary", "")
        retry_count = s.get("synthesis_retry_count", 0)
        max_retries = s.get("context_compress_retries", cfg.context_compress_retries)

        async def _synth_progress(done: int, total: int) -> None:
            await _emit({"type": "phase-update", "phase": "synthesize",
                   "message": f"Synthesizing reports... (deepening {done}/{total})"})

        s["report"] = await synthesize_report(
            s["user_query"], plan_text, s["subagent_reports"],
            failure_summary=failure_summary,
            output_language=s.get("output_language", "zh"),
            cancel_event=cancel_event,
            on_progress=_synth_progress,
        )

        # Retry if still truncated after continuation rounds
        if needs_continuation(s["report"]) and retry_count < max_retries and not failure_summary:
            from .synthesis import _generate_failure_summary
            summary = await _generate_failure_summary(
                user_query=s["user_query"],
                research_plan=plan_text,
                reports=s["subagent_reports"],
                partial_report=s["report"],
                reason="truncated after continuation rounds",
                chat_fn=chat,
            )
            s["synthesis_failure_summary"] = summary
            s["synthesis_retry_count"] = retry_count + 1

            s["report"] = await synthesize_report(
                s["user_query"], plan_text, s["subagent_reports"],
                failure_summary=summary,
                output_language=s.get("output_language", "zh"),
                cancel_event=cancel_event,
                on_progress=_synth_progress,
            )

        await _emit({"type": "report-draft", "content": s["report"][:1000],
               "report_length": len(s["report"])})
        await trace("synthesize", "node_exit", "Synthesis complete", {"report_length": len(s["report"])})

        pct = _progress(completed_weight, PHASE_WEIGHTS["synthesize"])
        await _emit({"type": "progress", "phase": "synthesize", "percent": pct})
        completed_weight += PHASE_WEIGHTS["synthesize"]

        await persist_checkpoint(s["run_id"], "synthesis", s)
        return s

    async def _citation_node(s: ResearchState) -> ResearchState:
        current_phase.set("cite")
        nonlocal completed_weight
        await _check_cancelled()
        source_count = len(s.get("sources", []))
        await trace("cite", "node_enter", f"Adding citations from {source_count} sources", {"source_count": source_count})
        await _emit({"type": "phase-update", "phase": "cite",
               "message": f"Adding citations from {source_count} sources..."})

        if source_count == 0:
            s["cited_report"] = s.get("report", "")
            verification: dict[str, bool] = {}
        else:
            s["cited_report"], verification = await add_citations(
                s["report"], s.get("sources", []),
            )
        total_urls = len(verification)
        accessible = sum(1 for v in verification.values() if v)
        await _emit({"type": "citations-added",
               "cited_report_length": len(s["cited_report"]),
               "urls_checked": total_urls,
               "urls_accessible": accessible})
        await trace("cite", "node_exit", "Citations added", {"urls_checked": total_urls, "urls_accessible": accessible, "cited_report_length": len(s["cited_report"])})

        pct = _progress(completed_weight, PHASE_WEIGHTS["cite"])
        await _emit({"type": "progress", "phase": "cite", "percent": pct})
        completed_weight += PHASE_WEIGHTS["cite"]

        await persist_checkpoint(s["run_id"], "citation", s)
        return s

    def _should_continue(s: ResearchState) -> str:
        return "synthesize" if s.get("research_complete", False) else "subagents"

    # Build graph — 7 nodes
    graph = StateGraph(ResearchState)
    graph.add_node("init", _init_node)
    graph.add_node("plan", _plan_node)
    graph.add_node("split", _split_node)
    graph.add_node("subagents", _subagents_node)
    graph.add_node("reflection", _reflection_node)
    graph.add_node("synthesize", _synthesize_node)
    graph.add_node("cite", _citation_node)

    graph.set_entry_point("init")
    graph.add_edge("init", "plan")
    graph.add_edge("plan", "split")
    graph.add_edge("split", "subagents")
    graph.add_edge("subagents", "reflection")
    graph.add_conditional_edges("reflection", _should_continue, {
        "synthesize": "synthesize", "subagents": "subagents",
    })
    graph.add_edge("synthesize", "cite")
    graph.add_edge("cite", END)

    compiled = graph.compile()
    return await compiled.ainvoke(state)


def _progress(completed: int, current_weight: float, partial: float = 1.0) -> int:
    w = current_weight * partial
    return min(99, int((completed + w) / TOTAL_WEIGHT * 100))

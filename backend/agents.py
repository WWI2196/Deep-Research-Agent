"""Agent functions for each pipeline stage.

Every agent calls _chat() which transparently routes through
the configured LLM provider (Gemini, OpenAI, Claude, HuggingFace).

Improvements over baseline:
 - Adaptive query refinement: re-searches when initial scores are low
 - Source diversity enforcement: caps per-domain sources
 - Cross-verification: checks claims appear in multiple sources
 - Smart retry: failed subagents get retried with alternative queries
"""

import concurrent.futures
import json
import logging
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from langsmith import traceable

from .config import get_config
from .events import emit
from .prompts import (
    PLANNER_SYSTEM,
    SPLITTER_SYSTEM,
    SCALING_SYSTEM,
    REFLECTION_SYSTEM,
    SUBAGENT_REPORT,
    SYNTHESIS_SYSTEM,
    CITATION_SYSTEM,
    SOURCE_SCORING_SYSTEM,
)
from .providers import get_provider
from . import search as firecrawl

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────

def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return ""


def _clean_think_tags(content: str) -> str:
    if "<think>" in content and "</think>" in content:
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


def _extract_payload_from_result(result: Any) -> Optional[Any]:
    """Normalize Firecrawl extract responses across SDK/version shapes."""
    if result is None:
        return None

    if isinstance(result, dict):
        if result.get("data") is not None:
            return result.get("data")
        if result.get("results") is not None:
            return result.get("results")
        if result.get("content") is not None:
            return result.get("content")
        if result.get("markdown") is not None:
            return result.get("markdown")
        return None

    if isinstance(result, (list, str)):
        return result

    return None


def _pick_first_nonempty(item: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _normalize_search_item(item: Dict[str, Any], source_label: str) -> Optional[Dict[str, Any]]:
    """Normalize Firecrawl search result records across SDK versions/providers."""
    if not isinstance(item, dict):
        return None

    url = _pick_first_nonempty(
        item,
        ["url", "link", "sourceURL", "source_url", "href", "website", "canonical_url"],
    )
    if not url:
        return None

    title = _pick_first_nonempty(item, ["title", "name", "headline"]) or url
    description = _pick_first_nonempty(
        item,
        ["description", "snippet", "summary", "content", "markdown", "text"],
    )

    return {
        "title": title,
        "url": url,
        "description": description,
        "source": source_label,
    }


def _strip_subtask_tags(text: str) -> str:
    """Remove bracket tags like [subtask_id] from report headings and body."""
    # Remove [word_word_word] patterns typically used as subtask IDs
    text = re.sub(r'\[([a-z0-9_]+(?:_[a-z0-9_]+)+)\]\s*', '', text)
    # Remove standalone bracket tags at start of lines
    text = re.sub(r'^\s*\[[a-z0-9_]+\]\s*', '', text, flags=re.MULTILINE)
    return text


def _continue_if_truncated(
    report: str,
    user_query: str,
    context: str,
    *,
    end_marker: Optional[str] = None,
    max_rounds: int = 4,
) -> str:
    """Continue generation when output appears cut off or required end marker is missing."""
    if not report:
        return report

    def _has_clean_ending(text: str) -> bool:
        tail = text.rstrip()
        if not tail:
            return True
        if tail.endswith(("```", "***", "---", "___", "**")):
            return True
        return tail.endswith((".", "!", "?", ":", ")", "]", '"', "”", "'", "’"))

    def _needs_continuation(text: str) -> bool:
        if end_marker and end_marker not in text:
            return True

        if len(text) < 500:
            return False

        tail = text.rstrip()
        if _has_clean_ending(tail):
            return False

        # If it ends with an alphanumeric token and no terminal punctuation,
        # it's very likely cut by token limits.
        if tail and tail[-1].isalnum():
            return True

        # Conservative fallback for dangling connective endings.
        last_word_match = re.search(r"([A-Za-z]+)\W*$", tail)
        last_word = last_word_match.group(1).lower() if last_word_match else ""
        dangling_words = {
            "and", "the", "of", "in", "to", "a", "an", "or", "but",
            "for", "with", "that", "is", "are", "was", "were", "as",
        }
        return last_word in dangling_words

    round_idx = 0
    tail_chars = 3000
    while round_idx < max_rounds and _needs_continuation(report):
        round_idx += 1
        logger.info("Report appears truncated (round %s/%s), requesting continuation...", round_idx, max_rounds)
        emit({
            "type": "subagent-step",
            "subtask_id": "synthesis",
            "subtask_title": "Report Synthesis",
            "step": "continuing",
            "message": f"Report was truncated, continuing generation (pass {round_idx}/{max_rounds})...",
        })
        try:
            marker_instruction = (
                f" End with the exact marker {end_marker}."
                if end_marker
                else ""
            )
            continuation = _chat(
                role="coordinator",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are continuing a research report that was cut off. "
                            "Pick up EXACTLY where the text ends. Do not repeat previous content. "
                            "Do not add preamble text. Continue seamlessly and finish the remaining sections."
                            f"{marker_instruction} "
                            f"The report is about: {user_query}"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Continue this report from where it was cut off:\n\n...{report[-tail_chars:]}",
                    },
                ],
                max_tokens=8192,
            )
            if not continuation or len(continuation.strip()) < 20:
                break
            report = report.rstrip() + "\n\n" + continuation.strip()
        except Exception as e:
            logger.warning(f"Continuation failed: {e}")
            lower = str(e).lower()
            if any(token in lower for token in ["400", "too long", "max", "context"]):
                tail_chars = max(1000, int(tail_chars * 0.65))
                continue
            break

    if end_marker and end_marker not in report:
        try:
            logger.info("End marker still missing after continuation rounds; requesting compact finalization pass...")
            final_piece = _chat(
                role="coordinator",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are finalizing a report that ended early. "
                            "Continue from the exact end, do not repeat prior text, and finish cleanly. "
                            f"End with the exact marker {end_marker}."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Continue and finish from this tail:\n\n...{report[-1200:]}",
                    },
                ],
                max_tokens=4096,
            )
            if final_piece and len(final_piece.strip()) >= 20:
                report = report.rstrip() + "\n\n" + final_piece.strip()
        except Exception as e:
            logger.warning(f"Final compact continuation failed: {e}")

    if end_marker:
        report = report.replace(end_marker, "").rstrip()
    return report


# ── unified LLM call ────────────────────────────────────────────────

def _chat(
    role: str,
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
) -> str:
    """Call the LLM configured for *role* with observability & retries."""
    cfg = get_config().get_role(role)
    provider = get_provider(cfg.provider)
    temp = temperature if temperature is not None else cfg.temperature

    for attempt in range(max_retries):
        try:
            emit({
                "type": "llm-call-start",
                "model": cfg.model,
                "provider": cfg.provider,
                "role": role,
                "attempt": attempt + 1,
            })

            result = provider.chat(
                model=cfg.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens,
            )
            result = _clean_think_tags(result)

            emit({
                "type": "llm-call-end",
                "model": cfg.model,
                "provider": cfg.provider,
                "role": role,
                "output_length": len(result),
            })
            return result

        except Exception as exc:
            err = str(exc)
            emit({
                "type": "llm-call-error",
                "model": cfg.model,
                "provider": cfg.provider,
                "role": role,
                "error": err[:200],
                "attempt": attempt + 1,
            })
            # Fatal – don't retry
            lower = err.lower()
            if any(c in lower for c in ["401", "403", "invalid"]):
                raise
            # Transient – back off
            if attempt < max_retries - 1:
                time.sleep(2.0 * (2 ** attempt))
                continue
            raise

    raise RuntimeError(f"LLM call exhausted retries for [{role}]")  # never reached


# ── pipeline agents ─────────────────────────────────────────────────

@traceable(name="plan_research")
def generate_research_plan(user_query: str) -> str:
    return _chat(
        role="planner",
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": user_query},
        ],
    )


@traceable(name="split_subtasks")
def split_into_subtasks(research_plan: str) -> List[Dict[str, Any]]:
    prompt = research_plan + "\n\nReturn the Output as valid JSON."
    response = _chat(
        role="splitter",
        messages=[
            {"role": "system", "content": SPLITTER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.strip()
    if not content:
        raise ValueError("Empty response from task splitter.")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        extracted = _extract_json(content)
        if not extracted:
            raise
        payload = json.loads(extracted)
    return payload["subtasks"]


@traceable(name="compute_scaling")
def compute_scaling(user_query: str, research_plan: str) -> Dict[str, Any]:
    response = _chat(
        role="scaler",
        messages=[
            {"role": "system", "content": SCALING_SYSTEM},
            {"role": "user", "content": f"Query: {user_query}\n\nPlan:\n{research_plan}\n\nReturn valid JSON."},
        ],
        temperature=0.1,
    )
    content = response.strip()
    if not content:
        raise ValueError("Empty response from scaler.")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        extracted = _extract_json(content)
        if not extracted:
            raise
        return json.loads(extracted)


@traceable(name="generate_search_queries")
def generate_search_queries(subtask: Dict[str, Any]) -> List[str]:
    prompt = (
        "You are an expert search engine query optimizer. "
        "Generate 4-7 diverse web search queries for the provided subtask.\n"
        "Include a mix of broad, specific, natural-language, and entity-centric queries.\n"
        "Return as JSON: {\"queries\": [\"q1\", \"q2\", ...]}\n\n"
        f"Subtask: {subtask['title']}\n"
        f"Description: {subtask['description']}\n"
        f"Objective: {subtask.get('objective', '')}\n"
        f"Preferred Sources: {subtask.get('source_types', '')}"
    )
    response = _chat(
        role="subagent",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    try:
        payload = json.loads(_extract_json(response))
        queries = payload.get("queries", [])
        # Inject semantic modifiers for source types
        source_types = subtask.get("source_types", "")
        if isinstance(source_types, str):
            source_types = [s.strip() for s in source_types.split(",")]
        modifiers = []
        if any("academic" in s.lower() or "paper" in s.lower() for s in source_types):
            modifiers.extend(["research paper", "study"])
        if any("code" in s.lower() or "github" in s.lower() for s in source_types):
            modifiers.extend(["github", "source code"])
        if any("official" in s.lower() or "docs" in s.lower() for s in source_types):
            modifiers.extend(["documentation", "official guide"])
        final = list(queries)
        if modifiers:
            for q in queries[:2]:
                for m in modifiers[:2]:
                    if m not in q.lower():
                        final.append(f"{q} {m}")
        seen = set()
        deduped = []
        for q in final:
            if q not in seen:
                seen.add(q)
                deduped.append(q)
        return deduped[:10]
    except Exception:
        return [subtask["title"]]


def batch_evaluate_sources(
    sources: List[Dict[str, Any]],
    user_query: str,
) -> List[Dict[str, Any]]:
    if not sources:
        return []
    batch_size = 20
    scored: List[Dict[str, Any]] = []
    for i in range(0, len(sources), batch_size):
        batch = sources[i : i + batch_size]
        sources_text = ""
        for idx, s in enumerate(batch):
            snippet = s.get("description", "") or s.get("snippet", "")
            sources_text += (
                f"ID: {idx}\nURL: {s.get('url', '')}\n"
                f"Title: {s.get('title', '')}\nSnippet: {snippet[:300]}\n\n"
            )
        try:
            response = _chat(
                role="evaluator",
                messages=[
                    {"role": "system", "content": SOURCE_SCORING_SYSTEM.format(user_query=user_query)},
                    {"role": "user", "content": f"Evaluate these sources:\n\n{sources_text}"},
                ],
                temperature=0.1,
            )
            content = response.strip()
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                result = json.loads(_extract_json(content))
            evals = {item["id"]: item for item in result.get("evaluations", [])}
            for idx, src in enumerate(batch):
                ev = evals.get(idx)
                src["quality_score"] = float(ev["score"]) if ev else 0.3
                src["reasoning"] = ev.get("reason", "") if ev else ""
                scored.append(src)
        except Exception as e:
            logger.warning(f"Source evaluation failed: {e}")
            for src in batch:
                src["quality_score"] = 0.5
                scored.append(src)
    return scored


# ── Source diversity enforcement ────────────────────────────────────

def _enforce_source_diversity(
    sources: List[Dict[str, Any]],
    max_per_domain: int = 3,
) -> List[Dict[str, Any]]:
    """Cap sources from the same domain to ensure breadth."""
    domain_count: Counter = Counter()
    diverse: List[Dict[str, Any]] = []
    for s in sources:
        url = s.get("url", "")
        try:
            domain = urlparse(url).netloc.replace("www.", "")
        except Exception:
            domain = url
        if domain_count[domain] < max_per_domain:
            domain_count[domain] += 1
            diverse.append(s)
    return diverse


# ── Adaptive query refinement ───────────────────────────────────────

def _refine_queries_if_needed(
    subtask: Dict[str, Any],
    scored_sources: List[Dict[str, Any]],
    original_queries: List[str],
    user_query: str,
) -> List[str]:
    """Generate refined queries if initial results scored poorly."""
    if not scored_sources:
        return []
    avg_score = sum(s.get("quality_score", 0) for s in scored_sources) / len(scored_sources)
    high_quality = [s for s in scored_sources if s.get("quality_score", 0) >= 0.7]

    # If average score is decent or we have enough good sources, skip refinement
    if avg_score >= 0.5 or len(high_quality) >= 3:
        return []

    emit({
        "type": "subagent-step",
        "subtask_id": subtask.get("id", ""),
        "subtask_title": subtask.get("title", ""),
        "step": "refining-queries",
        "message": f"Low avg score ({avg_score:.2f}), refining search queries...",
    })

    prompt = (
        "The initial search returned low-quality results. Generate 3-4 refined, "
        "more specific search queries that target authoritative sources.\n\n"
        f"Original queries: {json.dumps(original_queries[:4])}\n"
        f"Topic: {subtask.get('title', '')}\n"
        f"Objective: {subtask.get('objective', '')}\n"
        f"Preferred sources: {subtask.get('source_types', 'academic, official')}\n\n"
        'Return JSON: {"queries": ["q1", "q2", ...]}'
    )
    try:
        response = _chat(role="subagent", messages=[{"role": "user", "content": prompt}], temperature=0.4)
        payload = json.loads(_extract_json(response))
        new_queries = payload.get("queries", [])
        # Deduplicate against originals
        existing = set(q.lower() for q in original_queries)
        return [q for q in new_queries if q.lower() not in existing][:4]
    except Exception:
        return []


@traceable(name="run_subagent")
def run_subagent(
    user_query: str,
    research_plan: str,
    subtask: Dict[str, Any],
    tool_budget: int,
) -> Dict[str, Any]:
    sid = subtask["id"]
    stitle = subtask["title"]

    # 1 — generate queries
    emit({"type": "subagent-step", "subtask_id": sid, "subtask_title": stitle, "step": "generating-queries", "message": f"Generating queries for: {stitle}"})
    queries = generate_search_queries(subtask)
    emit({"type": "subagent-queries", "subtask_id": sid, "subtask_title": stitle, "queries": queries, "count": len(queries)})

    sources_list: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []

    # 2 — parallel search
    search_queries = queries[: max(1, min(10, tool_budget))]
    emit({"type": "subagent-step", "subtask_id": sid, "subtask_title": stitle, "step": "searching", "message": f"Searching {len(search_queries)} queries..."})

    def _search(q):
        try:
            emit({"type": "subagent-search", "subtask_id": sid, "query": q, "status": "started"})
            result = firecrawl.search(q, limit=8)
            n = len(result.get("data", [])) if result else 0
            emit({"type": "subagent-search", "subtask_id": sid, "query": q, "status": "completed", "results_found": n})
            return result
        except Exception as e:
            emit({"type": "subagent-search", "subtask_id": sid, "query": q, "status": "failed", "error": str(e)[:100]})
            return None

    with concurrent.futures.ThreadPoolExecutor() as pool:
        search_results = list(pool.map(_search, search_queries))

    raw_candidates: List[Dict[str, Any]] = []
    for sr in search_results:
        if not sr or not sr.get("data"):
            continue
        for item in sr["data"]:
            normalized = _normalize_search_item(item, "search")
            if normalized:
                raw_candidates.append(normalized)

    # 3 — evaluate
    emit({"type": "subagent-step", "subtask_id": sid, "subtask_title": stitle, "step": "evaluating-sources", "message": f"Evaluating {len(raw_candidates)} candidates..."})
    scored = batch_evaluate_sources(raw_candidates, subtask.get("objective", user_query))
    scored.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

    # 3b — adaptive query refinement: if results are poor, re-search
    refined_queries = _refine_queries_if_needed(subtask, scored, queries, user_query)
    if refined_queries:
        emit({"type": "subagent-queries", "subtask_id": sid, "subtask_title": stitle, "queries": refined_queries, "count": len(refined_queries)})
        with concurrent.futures.ThreadPoolExecutor() as pool:
            refined_results = list(pool.map(_search, refined_queries))
        for sr in refined_results:
            if not sr or not sr.get("data"):
                continue
            for item in sr["data"]:
                normalized = _normalize_search_item(item, "refined-search")
                if normalized:
                    raw_candidates.append(normalized)
        # Re-evaluate all candidates
        scored = batch_evaluate_sources(raw_candidates, subtask.get("objective", user_query))
        scored.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

    # 3c — enforce source diversity (max 3 per domain)
    scored = _enforce_source_diversity(scored, max_per_domain=3)
    sources_list = scored

    unique_urls: List[str] = []
    filtered: List[Dict[str, Any]] = []
    for s in sources_list:
        u = s.get("url")
        if u and u not in unique_urls:
            unique_urls.append(u)
            filtered.append(s)

    if not filtered:
        seen_raw = set()
        for item in raw_candidates:
            raw_url = item.get("url")
            if raw_url and raw_url not in seen_raw:
                seen_raw.add(raw_url)
                filtered.append({
                    "url": raw_url,
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "quality_score": 0.2,
                    "source": item.get("source", "search"),
                })

    top_urls = unique_urls[: min(12, tool_budget)]
    if not top_urls:
        top_urls = [s.get("url") for s in filtered if s.get("url")][: min(12, tool_budget)]

    emit({
        "type": "subagent-sources-scored", "subtask_id": sid, "subtask_title": stitle,
        "total_candidates": len(raw_candidates), "unique_sources": len(filtered),
        "top_urls": top_urls,
        "top_scores": [{"url": s["url"], "title": s.get("title", ""), "score": s.get("quality_score", 0)} for s in filtered[:10]],
    })

    # 4 — extract
    emit({"type": "subagent-step", "subtask_id": sid, "subtask_title": stitle, "step": "extracting", "message": f"Extracting from {len(top_urls)} sources..."})

    source_by_url = {s.get("url"): s for s in filtered if s.get("url")}

    def _extract(url):
        try:
            emit({"type": "subagent-extract", "subtask_id": sid, "url": url, "status": "started"})
            result = firecrawl.extract([url], f"Extract key facts about: {stitle}")
            payload = _extract_payload_from_result(result)
            ok = payload is not None
            emit({"type": "subagent-extract", "subtask_id": sid, "url": url, "status": "completed" if ok else "no-data"})
            return url, payload
        except Exception as e:
            emit({"type": "subagent-extract", "subtask_id": sid, "url": url, "status": "failed", "error": str(e)[:100]})
            return url, None

    with concurrent.futures.ThreadPoolExecutor() as pool:
        extract_results = list(pool.map(_extract, top_urls))

    for url, payload in extract_results:
        if payload is not None:
            evidence.append({"url": url, "data": payload})
            continue

        # Fallback: preserve at least snippet-level evidence when extract yields no payload
        src = source_by_url.get(url, {})
        snippet = src.get("description") or src.get("title")
        if snippet:
            evidence.append({"url": url, "data": snippet})

    evidence_text = "\n\n".join(f"[From {e['url']}]: {e['data']}" for e in evidence)

    # 5 — write report
    emit({"type": "subagent-step", "subtask_id": sid, "subtask_title": stitle, "step": "writing-report", "message": f"Writing with {len(evidence)} evidence items...", "evidence_count": len(evidence)})

    report = _chat(
        role="subagent",
        messages=[
            {
                "role": "system",
                "content": SUBAGENT_REPORT.format(
                    user_query=user_query,
                    research_plan=research_plan,
                    subtask_id=sid,
                    subtask_title=stitle,
                    subtask_description=subtask.get("description", ""),
                    subtask_objective=subtask.get("objective", ""),
                    subtask_output_format=subtask.get("output_format", ""),
                    subtask_tool_guidance=subtask.get("tool_guidance", ""),
                    subtask_source_types=subtask.get("source_types", ""),
                    subtask_boundaries=subtask.get("boundaries", ""),
                ),
            },
            {"role": "user", "content": f"Evidence:\n{evidence_text}"},
        ],
    )

    emit({
        "type": "subagent-complete", "subtask_id": sid, "subtask_title": stitle,
        "report_length": len(report), "sources_count": len(filtered[:20]), "evidence_count": len(evidence),
    })

    return {"subtask_id": sid, "report": report, "sources": filtered[:20], "evidence_count": len(evidence)}


@traceable(name="synthesize_report")
def synthesize_report(user_query: str, research_plan: str, reports: List[str]) -> str:
    combined = "\n\n".join(reports)
    max_chars = 80000
    for attempt in range(3):
        try:
            truncated = combined[:max_chars] if len(combined) > max_chars else combined
            result = _chat(
                role="coordinator",
                messages=[{
                    "role": "system",
                    "content": SYNTHESIS_SYSTEM.format(
                        user_query=user_query,
                        research_plan=research_plan[:3000],
                        subagent_reports=truncated,
                    ) + "\n\nIMPORTANT: End your full final answer with this exact marker on its own line: <<END_OF_REPORT>>",
                }],
                max_tokens=8192,
            )
            # Check if report was cut off and continue if needed
            result = _continue_if_truncated(
                result,
                user_query,
                truncated,
                end_marker="<<END_OF_REPORT>>",
                max_rounds=5,
            )
            # Strip any remaining bracket tags like [task_id]
            result = _strip_subtask_tags(result)
            return result
        except Exception as e:
            lower = str(e).lower()
            if "400" in lower or "too long" in lower or "max" in lower:
                max_chars = int(max_chars * 0.6)
                continue
            raise
    return f"# Research Report: {user_query}\n\n{combined[:max_chars]}"


@traceable(name="citation_agent")
def add_citations(report: str, sources: List[Dict[str, Any]]) -> str:
    # Strip subtask tags before citation pass
    report = _strip_subtask_tags(report)
    def _format_sources(limit: int, desc_limit: int) -> str:
        lines: List[str] = []
        seen_urls = set()
        for src in sources:
            url = str(src.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            title = str(src.get("title") or "Source").strip()
            description = str(src.get("description") or "").replace("\n", " ").strip()
            if len(description) > desc_limit:
                description = description[:desc_limit].rstrip() + "..."
            lines.append(f"- {title} | {url} | {description}")
            if len(lines) >= limit:
                break
        return "\n".join(lines)

    attempt_plan = [
        (60000, 50, 320, 8192),
        (45000, 35, 240, 6144),
        (32000, 24, 180, 4096),
        (22000, 16, 140, 3072),
    ]

    for idx, (report_limit, source_limit, desc_limit, out_tokens) in enumerate(attempt_plan, start=1):
        try:
            truncated_report = report[:report_limit] if len(report) > report_limit else report
            sources_text = _format_sources(source_limit, desc_limit)
            if not sources_text:
                logger.warning("Citation pass skipped due to missing usable source URLs.")
                return report

            result = _chat(
                role="citation",
                messages=[{
                    "role": "system",
                    "content": CITATION_SYSTEM.format(report=truncated_report, sources=sources_text),
                }],
                temperature=0.1,
                max_tokens=out_tokens,
            )
            result = _continue_if_truncated(result, "citation pass", "", max_rounds=3)
            result = _strip_subtask_tags(result)
            return result
        except Exception as e:
            lower = str(e).lower()
            logger.warning(
                "Citation agent attempt %s/%s failed (report=%s chars, sources=%s): %s",
                idx,
                len(attempt_plan),
                report_limit,
                source_limit,
                e,
            )
            if any(token in lower for token in ["400", "too long", "max", "context"]):
                continue
            break

    logger.warning("Citation agent failed after retries; returning uncited synthesized report.")
    return report

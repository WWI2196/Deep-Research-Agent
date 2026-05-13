"""Subagent orchestration: ReAct Agent-driven search, evaluate, extract, report."""

import asyncio
import json
import logging
import re
from typing import Any

from .config import get_config
from .helpers import (
    enforce_source_diversity,
    enforce_source_type_quota,
    extract_json,
    generate_broader_queries,
    normalize_search_item,
    query_similarity,
)
from .llm import chat
from .prompts import SOURCE_EVALUATE, SUBAGENT_REPORT
from .tracing import trace

logger = logging.getLogger(__name__)

# ── query modifiers by source type ─────────────────────────────────

_SOURCE_MODIFIERS = {
    "academic": ["research paper", "study", "pdf"],
    "paper": ["research paper", "study"],
    "official": ["official", "documentation", ".gov"],
    "docs": ["documentation", "official guide"],
    "code": ["github", "source code", "repository"],
    "github": ["github", "source code"],
    "news": ["latest", "2025", "report"],
    "industry report": ["market report", "industry analysis", "2025"],
    "data": ["statistics", "dataset", "data analysis"],
}


def generate_search_queries(subtask: dict[str, Any]) -> list[str]:
    """Generate search queries from subtask keywords and source types — rules-based.

    Falls back to title-based single query if no keywords are present.
    """
    keywords = subtask.get("keywords", [])
    source_types = subtask.get("source_types", "")

    title = subtask.get("title", "")
    if not keywords:
        return [f"{title} 2025", f"{title} 2026", title]

    if isinstance(source_types, str):
        source_types_list = [s.strip().lower() for s in source_types.split(",")]
    else:
        source_types_list = [s.lower() for s in source_types]

    # Collect applicable modifiers
    modifiers: list[str] = []
    for st in source_types_list:
        for key, mods in _SOURCE_MODIFIERS.items():
            if key in st:
                modifiers.extend(mods)

    # Build queries: keyword × modifier combos
    queries: list[str] = []
    for kw in keywords[:5]:
        queries.append(kw)  # raw keyword
        for mod in modifiers[:2]:
            if mod not in kw.lower():
                queries.append(f"{kw} {mod}")

    # Exact deduplicate
    seen: set[str] = set()
    deduped: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped.append(q)

    # Fuzzy deduplicate: collapse near-identical queries
    fuzzy_deduped: list[str] = []
    for q in deduped:
        is_dup = any(
            query_similarity(q, existing) >= 0.85
            for existing in fuzzy_deduped
        )
        if not is_dup:
            fuzzy_deduped.append(q)

    # Fallback for very few keywords: prepend year-boosted title queries
    if len(keywords) < 2 and title:
        year_queries = [f"{title} 2025", f"{title} 2026"]
        if title not in fuzzy_deduped:
            year_queries.append(title)
        fuzzy_deduped = year_queries + fuzzy_deduped

    logger.info("Generated %d search queries (fuzzy dedup from %d) for subtask %s",
                len(fuzzy_deduped[:10]), len(deduped), subtask.get("id", "?"))
    return fuzzy_deduped[:10]


async def batch_evaluate_sources(
    sources: list[dict[str, Any]],
    user_query: str,
) -> list[dict[str, Any]]:
    """Evaluate source quality AND decide full-text-worthiness in one LLM call.

    Each source gets a quality_score (0.0-1.0) and a full_text flag.
    """
    if not sources:
        return []

    batch_size = 20
    scored: list[dict[str, Any]] = []
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
            response = await chat(
                role="evaluator",
                messages=[
                    {"role": "system", "content": SOURCE_EVALUATE.format(user_query=user_query)},
                    {"role": "user", "content": f"Evaluate these sources:\n\n{sources_text}"},
                ],
                temperature=0.1,
            )
            content = response.strip()
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                result = json.loads(extract_json(content))
            evals = {item["id"]: item for item in result.get("evaluations", [])}
            for idx, src in enumerate(batch):
                ev = evals.get(idx)
                raw = float(ev.get("normalized_score") or ev.get("score", 0.3)) if ev else 0.3
                src["quality_score"] = max(0.0, min(1.0, raw))
                src["full_text"] = bool(ev.get("full_text", False)) if ev else False
                src["reasoning"] = ev.get("reason", "") if ev else ""
                scored.append(src)
        except Exception:
            for src in batch:
                src["quality_score"] = 0.3
                src["full_text"] = False
                scored.append(src)
    return scored


async def _refine_queries_if_needed(
    subtask: dict[str, Any],
    scored_sources: list[dict[str, Any]],
    original_queries: list[str],
) -> list[str]:
    if not scored_sources:
        return []
    avg_score = sum(s.get("quality_score", 0) for s in scored_sources) / len(scored_sources)
    high_quality = [s for s in scored_sources if s.get("quality_score", 0) >= 0.7]
    if avg_score >= 0.5 or len(high_quality) >= 3:
        return []

    prompt = (
        "Initial search returned low-quality results. Generate 3-4 refined, "
        "more specific search queries targeting authoritative sources.\n\n"
        f"Original queries: {json.dumps(original_queries[:4])}\n"
        f"Topic: {subtask.get('title', '')}\n"
        f"Objective: {subtask.get('objective', '')}\n"
        f"Preferred sources: {subtask.get('source_types', 'academic, official')}\n\n"
        'Return JSON: {"queries": ["q1", "q2", ...]}'
    )
    try:
        response = await chat(role="subagent", messages=[{"role": "user", "content": prompt}], temperature=0.4)
        payload = json.loads(extract_json(response))
        new_queries = payload.get("queries", [])
        existing = {q.lower() for q in original_queries}
        return [q for q in new_queries if q.lower() not in existing][:4]
    except Exception:
        return []


async def _search_document_collections(
    collection_ids: list[str],
    subtask: dict[str, Any],
    user_query: str,
) -> dict[str, Any] | None:
    """Search private document collections using hybrid retrieval."""
    from .document_store import DocumentStore

    store = DocumentStore()
    query = subtask.get("objective") or subtask.get("title") or user_query
    await trace("subagents", "rag_search_start", f"Searching document collections", {
        "collection_ids": collection_ids,
        "subtask_id": subtask.get("id"),
        "subtask_title": subtask.get("title"),
        "query": query,
        "top_k": 12,
    }, level="debug")
    results = await store.query(collection_ids, query, top_k=6)
    if not results:
        await trace("subagents", "rag_results", "No document results", {"count": 0}, level="debug")
        return None

    await trace("subagents", "rag_results", f"Document search returned {len(results)} chunks", {
        "count": len(results),
        "chunks": [
            {"doc_name": r.get("doc_name"), "score": r.get("score"), "chunk_id": r.get("chunk_id")}
            for r in results[:5]
        ],
    }, level="debug")

    items: list[dict[str, Any]] = []
    for r in results:
        fp = r.get("file_path", "")
        items.append({
            "title": r["doc_name"],
            "url": f"file://{fp}" if fp else f"doc://{r['doc_id']}",
            "description": r["text"][:2000],
            "score": r["score"],
            "source": "document",
        })
    return {"data": items}


async def run_subagent(
    user_query: str,
    research_plan: str,
    subtask: dict[str, Any],
    tool_budget: int,
    query_cache: dict[str, list[dict[str, Any]]] | None = None,
    document_collections: list[str] | None = None,
    gap_instruction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """ReAct Agent-driven subagent implementation."""
    from .react_agent import run_react_agent
    from .tools import build_research_tools
    from .prompts import SUBAGENT_REACT_SYSTEM

    sid = subtask.get("id") or f"fallback-{hash(subtask.get('title', '')) % 10000}"
    stitle = subtask.get("title", "Untitled")
    if query_cache is None:
        query_cache = {}

    # Build tools bound to shared query_cache
    tools = build_research_tools(
        query_cache=query_cache,
        document_collections=document_collections,
    )

    # Override submit_report tool to force-inject correct subtask ID/title.
    # The LLM sometimes omits the id when constructing the subtask dict itself,
    # so we remove it from params_schema and inject it server-side.
    for t in tools:
        if t.name == "submit_report":
            from .tools import submit_report_tool

            async def _bound_submit_report(
                evidence, user_query,
                _sid=sid, _stitle=stitle, _subtask=subtask, _rp=research_plan,
                **_unused,
            ):
                return await submit_report_tool(
                    evidence=evidence,
                    subtask={
                        "id": _sid,
                        "title": _stitle,
                        "description": _subtask.get("description", ""),
                        "objective": _subtask.get("objective", ""),
                        "output_format": _subtask.get("output_format", "markdown"),
                        "source_types": _subtask.get("source_types", ""),
                        "boundaries": _subtask.get("boundaries", ""),
                    },
                    user_query=user_query,
                    research_plan=_rp,
                )

            t.params_schema = {"evidence": "list[dict]", "user_query": "str"}
            t.description = (
                "Submit the final report for this subtask. Provide all collected evidence. "
                "This should be called ONLY when you have gathered sufficient evidence and are ready to finalize."
            )
            t.fn = _bound_submit_report
            break

    # Build system prompt
    system_prompt = SUBAGENT_REACT_SYSTEM.format(
        subtask_id=sid,
        subtask_title=stitle,
        subtask_objective=subtask.get("objective", ""),
        subtask_description=subtask.get("description", ""),
        subtask_source_types=subtask.get("source_types", ""),
        subtask_boundaries=subtask.get("boundaries", ""),
    )

    # Build user prompt with gap instruction if present
    user_prompt_parts = [
        f"Global query: {user_query}",
        f"Research plan: {research_plan[:2000]}",
        f"Subtask: {stitle} ({sid})",
        f"Objective: {subtask.get('objective', '')}",
    ]
    if document_collections:
        user_prompt_parts.append(
            f"Document collections available: {document_collections}. "
            "Use document_hybrid_search to search these collections."
        )
    if gap_instruction:
        user_prompt_parts.append(
            f"\n[REFLECTION GAP INSTRUCTION]\n"
            f"Type: {gap_instruction.get('gap_type', 'missing_evidence')}\n"
            f"Description: {gap_instruction.get('description', '')}\n"
            f"Suggested queries: {gap_instruction.get('suggested_queries', [])}\n"
            "Please address this gap in your investigation."
        )
    user_prompt = "\n\n".join(user_prompt_parts)

    await trace("subagents", "react_start", f"Starting ReAct subagent {sid}", {
        "subtask_id": sid,
        "tool_budget": tool_budget,
        "has_gap_instruction": gap_instruction is not None,
    }, level="debug")

    # Decouple max_steps from search budget: give enough room for
    # search → evaluate → fetch → synthesize → write → submit flow.
    max_steps = min(max(tool_budget * 2, 12), 18)
    result = await run_react_agent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tools=tools,
        chat_fn=chat,
        max_steps=max_steps,
        temperature=0.3,
        subtask_id=sid,
    )

    report = result.get("final_answer", "")
    tool_calls = result.get("tool_calls", [])

    # Compress overly long reports by keeping high-importance paragraphs
    if len(report) > 8000:
        paragraphs = [p for p in report.split("\n\n") if p.strip()]
        if len(paragraphs) > 3:
            scored: list[tuple[int, str, int]] = []
            for i, para in enumerate(paragraphs):
                score = 0
                if "[src:" in para:
                    score += 3
                if re.search(r"\b\d{4}\b|\b\d+\.\d+|```|\bdef\s+\w+|\bclass\s+\w+", para):
                    score += 2
                if i < 2:
                    score += 1
                scored.append((i, para, score))
            scored.sort(key=lambda x: x[2], reverse=True)
            # Keep top paragraphs up to ~6000 chars
            kept: list[tuple[int, str]] = []
            current_len = 0
            for idx, para, _ in scored:
                para_len = len(para) + 2
                if current_len + para_len <= 6000:
                    kept.append((idx, para))
                    current_len += para_len
            kept.sort(key=lambda x: x[0])
            report = "\n\n".join(p for _, p in kept)

    # Fallback for empty report
    if len(report) < 200:
        report = (
            f"# {stitle}\n\n## Summary\n\n"
            f"Research on this subtask was not completed. Tool calls: {len(tool_calls)}.\n\n## Sources\n\n"
            + "\n".join(
                f"- {tc['input'].get('query', tc['input'].get('urls', ['unknown'])[0])}"
                for tc in tool_calls[:10]
                if tc.get("input")
            )
        )

    # Extract sources from tool calls for backward compatibility
    sources = _extract_sources_from_tool_calls(tool_calls)

    await trace("subagents", "react_end", f"ReAct subagent {sid} complete", {
        "subtask_id": sid,
        "report_length": len(report),
        "steps_taken": result.get("steps_taken", 0),
        "sources_count": len(sources),
    }, level="debug")

    return {
        "subtask_id": sid,
        "subtask_title": stitle,
        "report": report,
        "sources": sources,
        "evidence_count": len([tc for tc in tool_calls if tc.get("tool") in ("searxng_search", "document_hybrid_search")]),
        "tool_calls": tool_calls,
    }


def _extract_sources_from_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract unique sources from search tool calls for backward compatibility."""
    sources: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for tc in tool_calls:
        if tc.get("tool") not in ("searxng_search", "document_hybrid_search"):
            continue
        result = tc.get("result", {})
        if not result.get("success"):
            continue
        data = result.get("result", {})
        for item in data.get("results", []):
            url = item.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                source_type = item.get("source", "search")
                raw_score = item.get("quality_score")
                if raw_score is None:
                    raw_score = item.get("score")
                try:
                    raw_score = float(raw_score) if raw_score is not None else None
                except (ValueError, TypeError):
                    raw_score = None
                if source_type == "document":
                    quality_score = max(raw_score or 0.0, 0.85)
                else:
                    quality_score = raw_score if raw_score is not None else 0.5
                sources.append({
                    "url": url,
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "quality_score": quality_score,
                    "source": source_type,
                    "doc_id": item.get("doc_id", ""),
                })
    return sources


async def run_subagents_parallel(
    user_query: str,
    research_plan: str,
    subtasks: list[dict[str, Any]],
    tool_calls_per_subagent: int,
    query_cache: dict[str, list[dict[str, Any]]] | None = None,
    document_collections: list[str] | None = None,
    gap_instructions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if query_cache is None:
        query_cache = {}
    if gap_instructions is None:
        gap_instructions = []

    tasks = []
    for st in subtasks:
        gap = next(
            (g for g in gap_instructions if g.get("target_subtask_id") == st["id"]),
            None,
        )
        tasks.append(run_subagent(
            user_query, research_plan, st, tool_calls_per_subagent,
            query_cache=query_cache,
            document_collections=document_collections,
            gap_instruction=gap,
        ))
    results = await asyncio.gather(*tasks, return_exceptions=True)

    reports, sources, successful = [], [], []
    for r in results:
        if isinstance(r, Exception):
            logger.warning("Subagent failed: %s", r)
            continue
        reports.append(r["report"])
        sources.extend(r.get("sources", []))
        successful.append(r)

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for s in sources:
        u = s.get("url")
        if u and u not in seen:
            seen.add(u)
            unique.append(s)

    return {
        "reports": reports,
        "sources": unique,
        "raw": successful,
        "success_count": len(successful),
        "total_count": len(subtasks),
    }

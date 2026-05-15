"""Research tools for ReAct Agent.

Each tool is a pure async function that receives parameters, executes, and returns
a structured result. Tool internals preserve existing engineering optimizations
(query cache, empty-result rollback, batch evaluation, source diversity, etc.).

Design principle: LLM decides WHAT to do; code decides HOW to do it.
"""

import asyncio
import json
import logging
from typing import Any, Callable

from . import search as search_mod
from .config import get_config
from .helpers import (
    enforce_source_diversity,
    enforce_source_type_quota,
    filter_search_results,
    generate_broader_queries,
    normalize_search_item,
)
from .llm import chat
from .prompts import SOURCE_EVALUATE, SUBAGENT_REPORT
from .tracing import trace

logger = logging.getLogger(__name__)


class Tool:
    """Lightweight tool wrapper for ReAct Agent."""

    def __init__(
        self,
        name: str,
        description: str,
        params_schema: dict[str, str],
        fn: Callable,
    ):
        self.name = name
        self.description = description
        self.params_schema = params_schema
        self.fn = fn

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        try:
            result = await self.fn(**kwargs)
            return {"success": True, "result": result, "error": None}
        except Exception as exc:
            logger.warning("Tool %s failed: %s", self.name, exc)
            return {"success": False, "result": None, "error": str(exc)}


# ── searxng_search tool ───────────────────────────────────────────

async def searxng_search_tool(
    query: str,
    limit: int = 8,
    query_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Search the web using SearXNG.

    Uses query_cache to avoid duplicate searches across subagents.
    On empty results, automatically falls back to broader queries.
    """
    if query_cache is None:
        query_cache = {}

    await trace("subagents", "tool_call_start", f"searxng_search: {query[:60]}", {
        "query": query,
        "limit": limit,
        "cache_hit": query in query_cache,
    }, level="debug")

    if query in query_cache:
        logger.info("Query cache HIT for '%s'", query[:60])
        cached = query_cache[query]
        await trace("subagents", "tool_call_end", "searxng_search cache hit", {
            "query": query,
            "results": len(cached.get("data", [])),
        }, level="debug")
        return {"query": query, "results": cached.get("data", []), "source": "searxng", "cached": True}

    result = await search_mod.search(query, limit=limit)
    query_cache[query] = result if result else {"data": []}

    results_data: list[dict[str, Any]] = []
    if result and result.get("data"):
        for item in result["data"]:
            normalized = normalize_search_item(item, "searxng")
            if normalized:
                results_data.append(normalized)
        # Filter out irrelevant results
        before_filter = len(results_data)
        results_data = filter_search_results(results_data, query)
        if before_filter != len(results_data):
            logger.info("Filtered %d irrelevant results for query '%s'", before_filter - len(results_data), query[:60])

    # Empty-result rollback: try broader queries
    if not results_data:
        broader_queries = generate_broader_queries(query)
        for bq in broader_queries:
            if bq in query_cache:
                bq_result = query_cache[bq]
            else:
                bq_result = await search_mod.search(bq, limit=limit)
                query_cache[bq] = bq_result if bq_result else {"data": []}
            if bq_result and bq_result.get("data"):
                for item in bq_result["data"]:
                    normalized = normalize_search_item(item, "searxng")
                    if normalized:
                        results_data.append(normalized)
                results_data = filter_search_results(results_data, bq)
                if results_data:
                    logger.info("Empty-result rollback: broader '%s' -> %d results", bq, len(results_data))
                    break

    await trace("subagents", "tool_call_end", f"searxng_search complete ({len(results_data)} results)", {
        "query": query,
        "results": len(results_data),
        "rollback_used": len(results_data) > 0 and not (result and result.get("data")),
    }, level="debug")

    return {"query": query, "results": results_data, "source": "searxng", "cached": False}


# ── document_hybrid_search tool ───────────────────────────────────

async def document_hybrid_search_tool(
    query: str,
    collection_ids: list[str],
    top_k: int = 6,
) -> dict[str, Any]:
    """Search private document collections using hybrid retrieval (vector + BM25 + RRF).

    Document sources are highly trusted and marked for full-text usage.
    """
    from .document_store import get_document_store

    store = get_document_store()
    results = await store.query(collection_ids, query, top_k=top_k)

    items: list[dict[str, Any]] = []
    for r in results:
        fp = r.get("file_path", "")
        items.append({
            "title": r["doc_name"],
            "url": f"file://{fp}" if fp else f"doc://{r['doc_id']}",
            "description": r["text"][:2000],
            "score": r["score"],
            "source": "document",
            "full_text": True,
        })

    await trace("subagents", "tool_call_end", f"document_hybrid_search complete ({len(items)} results)", {
        "query": query,
        "results": len(items),
    }, level="debug")

    return {"query": query, "results": items, "source": "document"}


# ── evaluate_sources tool ─────────────────────────────────────────

async def evaluate_sources_tool(
    candidates: list[dict[str, Any]],
    objective: str,
    max_per_domain: int | None = None,
) -> dict[str, Any]:
    """Evaluate source quality and select which ones deserve full-text extraction.

    Returns scored sources + a list of URLs selected for full-text reading.
    Internally uses batch LLM evaluation + source diversity enforcement.
    """

    if max_per_domain is None:
        max_per_domain = get_config().max_sources_per_domain

    if not candidates:
        return {"scored": [], "selected_for_fulltext": []}

    await trace("subagents", "tool_call_start", f"evaluate_sources: {len(candidates)} candidates", {
        "count": len(candidates),
        "objective": objective[:100],
    }, level="debug")

    # Deduplicate by URL before evaluation
    seen_urls: set[str] = set()
    unique_candidates: list[dict[str, Any]] = []
    for c in candidates:
        url = c.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_candidates.append(c)

    # Batch LLM evaluation — all batches in parallel
    from .helpers import extract_json

    batch_size = 20

    async def _eval_batch(batch: list[dict[str, Any]], batch_idx: int) -> list[dict[str, Any]]:
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
                    {"role": "system", "content": SOURCE_EVALUATE.format(user_query=objective)},
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
            return batch
        except Exception as exc:
            logger.warning("Batch evaluate failed for batch %d: %s", batch_idx, exc)
            for src in batch:
                src["quality_score"] = 0.3
                src["full_text"] = False
            return batch

    batch_tasks = [
        _eval_batch(unique_candidates[i:i + batch_size], i // batch_size)
        for i in range(0, len(unique_candidates), batch_size)
    ]
    batch_results = await asyncio.gather(*batch_tasks)
    scored: list[dict[str, Any]] = [src for batch in batch_results for src in batch]

    # Document sources are highly trusted
    for s in scored:
        if s.get("source") == "document":
            s["quality_score"] = max(s.get("quality_score", 0), 0.85)
            s["full_text"] = True

    scored.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

    # Enforce diversity and quotas
    cfg = get_config()
    scored = enforce_source_diversity(scored, max_per_domain=max_per_domain)
    scored = enforce_source_type_quota(
        scored,
        quotas=getattr(cfg, "source_type_quotas", None),
        min_per_type=getattr(cfg, "min_source_per_type", None),
    )

    selected_for_fulltext = [
        s["url"] for s in scored
        if s.get("full_text") and s.get("url")
    ]
    # Ensure at least top 5 by score if LLM didn't select enough
    if len(selected_for_fulltext) < 3:
        for s in scored[:5]:
            if s.get("url"):
                selected_for_fulltext.append(s["url"])
        selected_for_fulltext = list(dict.fromkeys(selected_for_fulltext))

    await trace("subagents", "tool_call_end", f"evaluate_sources complete ({len(scored)} scored, {len(selected_for_fulltext)} fulltext)", {
        "scored": len(scored),
        "selected_for_fulltext": len(selected_for_fulltext),
    }, level="debug")

    return {
        "scored": scored,
        "selected_for_fulltext": selected_for_fulltext,
    }


# ── fetch_fulltext tool ───────────────────────────────────────────

async def fetch_fulltext_tool(urls: list[str]) -> dict[str, Any]:
    """Fetch full-text content from URLs via trafilatura (fast) or Crawl4AI (fallback).

    Skips file:// URLs (document library sources already have full text).
    Returns a map of URL -> extracted markdown.
    """
    await trace("subagents", "tool_call_start", f"fetch_fulltext: {len(urls)} urls", {
        "urls": urls[:10],
    }, level="debug")

    skipped: list[str] = []

    async def _extract_one(url: str) -> tuple[str, str | None]:
        if url.startswith("file://"):
            return url, None
        try:
            text = await search_mod.extract_async(url)
            return url, text
        except Exception as exc:
            logger.warning("Extract failed for %s: %s", url[:60], exc)
            return url, None

    tasks = [_extract_one(url) for url in urls[:12]]
    results = await asyncio.gather(*tasks)
    extracted = {url: text for url, text in results if text}

    await trace("subagents", "tool_call_end", f"fetch_fulltext complete ({len(extracted)} succeeded, {len(skipped)} skipped)", {
        "requested": len(urls),
        "succeeded": len(extracted),
        "skipped": len(skipped),
    }, level="debug")

    return {"extracted": extracted, "failed_count": len(urls) - len(extracted) - len(skipped), "skipped": skipped}


# ── synthesize_evidence tool ──────────────────────────────────────

async def synthesize_evidence_tool(
    findings: str,
    key_entities: list[str] | None = None,
    remaining_questions: list[str] | None = None,
    proposed_next_queries: list[str] | None = None,
) -> dict[str, Any]:
    """Synthesize evidence from previous searches to form intermediate conclusions.

    This tool enables multi-hop reasoning by forcing the agent to explicitly
    reflect on what has been found, extract key entities, identify gaps,
    and plan follow-up queries before continuing.

    No external API is called — the reasoning is recorded and returned
    as structured output for the agent's own context.
    """
    await trace("subagents", "tool_call_start", "synthesize_evidence", {
        "findings_length": len(findings),
        "entities": key_entities or [],
        "remaining_questions": remaining_questions or [],
        "proposed_queries": proposed_next_queries or [],
    }, level="debug")

    # Build a structured synthesis record that the LLM can reference later
    synthesis = {
        "findings": findings,
        "key_entities": key_entities or [],
        "remaining_questions": remaining_questions or [],
        "proposed_next_queries": proposed_next_queries or [],
        "hop_number": 0,  # Will be updated by react_agent if tracking hops
    }

    await trace("subagents", "tool_call_end", "synthesize_evidence complete", {
        "entities_count": len(key_entities or []),
        "proposed_queries_count": len(proposed_next_queries or []),
    }, level="debug")

    return {
        "synthesis": synthesis,
        "message": (
            "Evidence synthesized. Key entities and remaining questions recorded. "
            "Use the proposed_next_queries (or refine them) for your next search step."
        ),
    }


# ── submit_report tool ────────────────────────────────────────────

async def submit_report_tool(
    evidence: list[dict[str, Any]],
    subtask: dict[str, Any],
    user_query: str,
    research_plan: str = "",
) -> dict[str, Any]:
    """Generate the final subagent report from collected evidence.

    Retry once if report is too short or missing citations.
    """
    sid = subtask.get("id") or f"fallback-{hash(subtask.get('title', '')) % 10000}"
    stitle = subtask.get("title", "Untitled")

    await trace("subagents", "tool_call_start", f"submit_report: {stitle}", {
        "subtask_id": sid,
        "evidence_count": len(evidence),
    }, level="debug")

    evidence_text = "\n\n".join(
        f"[From {e.get('url', 'unknown')}]: {e.get('data', e.get('text', e.get('content', str(e))))}"
        for e in evidence
    )

    report = ""
    for attempt in range(2):
        retry_hint = ""
        if attempt == 1:
            hints = []
            if len(report) < 200:
                hints.append("YOUR PREVIOUS RESPONSE WAS EMPTY OR TOO SHORT.")
            if "[src:" not in report:
                hints.append(
                    "YOUR PREVIOUS RESPONSE HAD NO [src: url] CITATIONS. "
                    "Every factual claim MUST be followed by [src: <url>] immediately."
                )
            if hints:
                retry_hint = "\n\n" + " ".join(hints) + " Write a complete 800-1500 word report with proper citations."

        report = await chat(
            role="subagent",
            messages=[{
                "role": "system",
                "content": SUBAGENT_REPORT.format(
                    user_query=user_query,
                    research_plan=research_plan[:3000],
                    subtask_id=sid,
                    subtask_title=stitle,
                    subtask_description=subtask.get("description", ""),
                    subtask_objective=subtask.get("objective", ""),
                    subtask_output_format=subtask.get("output_format", "markdown"),
                    subtask_tool_guidance=subtask.get("tool_guidance", ""),
                    subtask_source_types=subtask.get("source_types", ""),
                    subtask_boundaries=subtask.get("boundaries", ""),
                ),
            }, {
                "role": "user",
                "content": f"Evidence:\n{evidence_text}" + retry_hint,
            }],
        )
        if len(report) >= 200 and "[src:" in report:
            break

    if len(report) < 200:
        report = (
            f"# {stitle}\n\n## Summary\n\n"
            f"Research on this subtask was not completed. Evidence collected: {len(evidence)} sources.\n\n## Sources\n\n"
            + "\n".join(f"- [{e.get('url', 'unknown')}]({e.get('url', '#')})" for e in evidence[:10])
        )

    await trace("subagents", "tool_call_end", f"submit_report complete ({len(report)} chars)", {
        "subtask_id": sid,
        "report_length": len(report),
        "attempts": attempt + 1,
    }, level="debug")

    return {"report": report}


# ── Tool registry ─────────────────────────────────────────────────

def build_research_tools(
    query_cache: dict[str, dict[str, Any]] | None = None,
    document_collections: list[str] | None = None,
) -> list[Tool]:
    """Build tool instances bound to shared query_cache."""
    if query_cache is None:
        query_cache = {}

    async def _searxng_search_wrapped(**kwargs: Any) -> dict[str, Any]:
        return await searxng_search_tool(query_cache=query_cache, **kwargs)

    tools: list[Tool] = [
        Tool(
            name="searxng_search",
            description=(
                "Search the web using SearXNG. Returns a list of search results with "
                "title, url, description, and score. Use this to find web pages, news, "
                "academic papers, and official documents. Supports empty-result rollback."
            ),
            params_schema={"query": "str", "limit": "int (optional, default 8)"},
            fn=_searxng_search_wrapped,
        ),
    ]
    if document_collections:
        tools.append(
            Tool(
                name="document_hybrid_search",
                description=(
                    "Search private document collections using hybrid retrieval "
                    "(Chroma vector + bm25s keyword + RRF fusion). Returns document chunks "
                    "with full text already available. Highly trusted source."
                ),
                params_schema={"query": "str", "collection_ids": "list[str]", "top_k": "int (optional, default 12)"},
                fn=document_hybrid_search_tool,
            )
        )
    tools.extend([
        Tool(
            name="evaluate_sources",
            description=(
                "Evaluate a batch of candidate sources for quality and relevance. "
                "Returns scored sources + a list of URLs recommended for full-text extraction. "
                "Automatically enforces source diversity and type quotas."
            ),
            params_schema={"candidates": "list[dict]", "objective": "str"},
            fn=evaluate_sources_tool,
        ),
        Tool(
            name="fetch_fulltext",
            description=(
                "Fetch full-text content from URLs via trafilatura. "
                "Skips file:// URLs (document sources already have full text). "
                "Returns extracted markdown per URL."
            ),
            params_schema={"urls": "list[str]"},
            fn=fetch_fulltext_tool,
        ),
        Tool(
            name="synthesize_evidence",
            description=(
                "Synthesize evidence from previous searches to form intermediate conclusions, "
                "identify gaps, and plan next steps. Use this BETWEEN searches to reason about "
                "what you have found and what additional information is needed. This enables "
                "multi-hop reasoning by forcing explicit reflection before the next retrieval step. "
                "Provide: findings summary, key_entities extracted, remaining_questions, and "
                "proposed_next_queries for follow-up searches."
            ),
            params_schema={
                "findings": "str",
                "key_entities": "list[str] (optional)",
                "remaining_questions": "list[str] (optional)",
                "proposed_next_queries": "list[str] (optional)",
            },
            fn=synthesize_evidence_tool,
        ),
        Tool(
            name="submit_report",
            description=(
                "Submit the final report for this subtask. Provide all collected evidence "
                "and the subtask metadata. This should be called ONLY when you have gathered "
                "sufficient evidence and are ready to finalize."
            ),
            params_schema={"evidence": "list[dict]", "subtask": "dict", "user_query": "str"},
            fn=submit_report_tool,
        ),
    ])
    return tools

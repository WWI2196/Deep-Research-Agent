"""Subagent orchestration: search, evaluate, extract, report."""

import asyncio
import json
import logging
from typing import Any

from . import search as search_mod
from .config import get_config
from .helpers import (
    enforce_source_diversity,
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

    if not keywords:
        return [subtask["title"]]

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
                src["quality_score"] = float(ev["score"]) if ev else 0.3
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
    results = await store.query(collection_ids, query, top_k=12)
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
) -> dict[str, Any]:
    sid = subtask["id"]
    stitle = subtask["title"]
    if query_cache is None:
        query_cache = {}

    # 1 — generate queries (rules-based, no LLM call)
    queries = generate_search_queries(subtask)

    # 2 — parallel search with query cache + empty-result rollback
    search_queries = queries[: max(1, min(10, tool_budget))]

    async def _search_one_cached(q: str) -> dict[str, Any] | None:
        if q in query_cache:
            logger.info("Query cache HIT for '%s' (subtask %s)", q[:60], sid)
            return query_cache[q]
        try:
            result = await asyncio.to_thread(search_mod.search, q, limit=8)
            query_cache[q] = result if result else {"data": []}

            if result is not None and not result.get("data"):
                broader = generate_broader_queries(q)
                for bq in broader:
                    if bq not in query_cache:
                        bq_result = await asyncio.to_thread(
                            search_mod.search, bq, limit=8
                        )
                        query_cache[bq] = bq_result if bq_result else {"data": []}
                        if bq_result and bq_result.get("data"):
                            logger.info(
                                "Empty-result rollback: broader '%s' -> %d results",
                                bq, len(bq_result["data"]),
                            )
                            return bq_result
            return result
        except Exception:
            return None

    tasks: list[Any] = [_search_one_cached(q) for q in search_queries]
    if document_collections:
        tasks.append(_search_document_collections(document_collections, subtask, user_query))
    search_results = await asyncio.gather(*tasks)

    raw_candidates: list[dict[str, Any]] = []
    for sr in search_results:
        if not sr or not sr.get("data"):
            continue
        for item in sr["data"]:
            normalized = normalize_search_item(item, "search")
            if normalized:
                raw_candidates.append(normalized)

    # 3 — evaluate + select full-text in one LLM call
    scored = await batch_evaluate_sources(raw_candidates, subtask.get("objective", user_query))
    scored.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

    # Document sources are highly trusted and always treated as full-text
    for s in scored:
        if s.get("source") == "document":
            s["quality_score"] = max(s.get("quality_score", 0), 0.85)
            s["full_text"] = True

    # 3b — adaptive query refinement
    refined_queries = await _refine_queries_if_needed(subtask, scored, queries)
    if refined_queries:
        refined_tasks = [_search_one_cached(q) for q in refined_queries]
        refined_results = await asyncio.gather(*refined_tasks)
        for sr in refined_results:
            if not sr or not sr.get("data"):
                continue
            for item in sr["data"]:
                normalized = normalize_search_item(item, "refined-search")
                if normalized:
                    raw_candidates.append(normalized)
        scored = await batch_evaluate_sources(raw_candidates, subtask.get("objective", user_query))
        scored.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

    # 3c — enforce source diversity
    scored = enforce_source_diversity(scored, max_per_domain=3)

    # Deduplicate
    seen_urls: set[str] = set()
    filtered: list[dict[str, Any]] = []
    for s in scored:
        u = s.get("url")
        if u and u not in seen_urls:
            seen_urls.add(u)
            filtered.append(s)

    if not filtered:
        for item in raw_candidates:
            raw_url = item.get("url")
            if raw_url and raw_url not in seen_urls:
                seen_urls.add(raw_url)
                filtered.append({
                    "url": raw_url,
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "quality_score": 0.2,
                    "source": item.get("source", "search"),
                })

    # Determine which URLs to fetch full-text (from LLM evaluation)
    full_text_urls: set[str] = {
        s["url"] for s in filtered
        if s.get("full_text") and s.get("url")
    }
    # Ensure at least top 5 by score if LLM didn't select enough
    if len(full_text_urls) < 3:
        for s in filtered[:5]:
            if s.get("url"):
                full_text_urls.add(s["url"])

    top_urls = [s.get("url") for s in filtered if s.get("url")][: min(12, tool_budget)]
    if not top_urls:
        filtered = [{"url": rc.get("url"), "title": rc.get("title", ""),
                     "description": rc.get("description", ""), "quality_score": 0.2}
                    for rc in raw_candidates if rc.get("url") and rc.get("url") not in seen_urls]
        seen_urls.update(s.get("url") for s in filtered)
        top_urls = [s.get("url") for s in filtered if s.get("url")][: min(12, tool_budget)]

    # 4 — extract full text (markdown) via trafilatura for full_text_urls
    async def _extract_one(url: str) -> tuple[str, str | None]:
        try:
            if url.startswith("file://"):
                return url, None
            if url in full_text_urls:
                text = await asyncio.to_thread(search_mod.extract, url)
                return url, text
            return url, None
        except Exception:
            return url, None

    extract_tasks = [_extract_one(url) for url in top_urls[:12]]
    extract_results = await asyncio.gather(*extract_tasks)
    extracted_map = {url: text for url, text in extract_results if text}

    # 5 — build evidence
    evidence: list[dict[str, Any]] = []
    for s in filtered:
        url = s.get("url")
        if not url or url not in top_urls:
            continue
        if s.get("source") == "document":
            text = s.get("description", "")
            if text:
                evidence.append({
                    "url": url, "data": f"[DOCUMENT] {text}",
                    "score": s.get("quality_score", 0.85),
                })
        elif url in extracted_map:
            evidence.append({
                "url": url, "data": f"[FULL-TEXT] {extracted_map[url]}",
                "score": s.get("quality_score", 0.2),
            })
        else:
            snippet = s.get("description") or s.get("title", "")
            if snippet:
                evidence.append({
                    "url": url, "data": f"[SNIPPET] {snippet}",
                    "score": s.get("quality_score", 0.2),
                })

    # Compress: keep only top N tool results by quality score
    cfg = get_config()
    keep_n = cfg.keep_tool_results
    evidence_before = len(evidence)
    if keep_n > 0 and len(evidence) > keep_n:
        evidence.sort(key=lambda x: x.get("score", 0.2), reverse=True)
        evidence = evidence[:keep_n]
        logger.info("Compressed evidence from %d to %d items (keep_tool_results=%d)",
                     evidence_before, keep_n, keep_n)

    await trace("subagents", "evidence_built", f"Built {len(evidence)} evidence items", {
        "subtask_id": sid,
        "evidence_count": len(evidence),
        "compressed_from": evidence_before if keep_n > 0 and evidence_before > keep_n else None,
        "keep_tool_results": keep_n,
        "sources": [
            {"url": e["url"], "score": e.get("score"), "type": e["data"].split("]")[0].lstrip("[")}
            for e in evidence[:10]
        ],
    }, level="debug")

    # Truncate each evidence item to keep context manageable
    max_evidence_per_item = 3000
    for e in evidence:
        if len(e["data"]) > max_evidence_per_item:
            e["data"] = e["data"][:max_evidence_per_item] + "\n...[truncated]"

    evidence_text = "\n\n".join(f"[From {e['url']}]: {e['data']}" for e in evidence)

    def _retry_hint(attempt: int, prev_report: str) -> str:
        if attempt == 0:
            return ""
        hints = []
        if len(prev_report) < 200:
            hints.append("YOUR PREVIOUS RESPONSE WAS EMPTY OR TOO SHORT.")
        if "[src:" not in prev_report:
            hints.append("YOUR PREVIOUS RESPONSE HAD NO [src: url] CITATIONS. "
                         "Every factual claim MUST be followed by [src: <url>] immediately.")
        if not hints:
            return ""
        return "\n\n" + " ".join(hints) + " Write a complete 800-1500 word report with proper citations."

    # 6 — write report (retry once if too short or missing citations)
    report = ""
    for attempt in range(2):
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
                "content": f"Evidence:\n{evidence_text}" + _retry_hint(attempt, report),
            }],
        )
        if len(report) >= 200 and "[src:" in report:
            break
    if len(report) < 200:
        report = f"# {stitle}\n\n## Summary\n\nResearch on this subtask was not completed. Evidence collected: {len(evidence)} sources.\n\n## Sources\n\n" + "\n".join(
            f"- [{e['url']}]({e['url']})" for e in evidence[:10]
        )

    return {
        "subtask_id": sid,
        "subtask_title": stitle,
        "report": report,
        "sources": filtered[:20],
        "evidence_count": len(evidence),
    }


async def run_subagents_parallel(
    user_query: str,
    research_plan: str,
    subtasks: list[dict[str, Any]],
    tool_calls_per_subagent: int,
    query_cache: dict[str, list[dict[str, Any]]] | None = None,
    document_collections: list[str] | None = None,
) -> dict[str, Any]:
    if query_cache is None:
        query_cache = {}

    tasks = [
        run_subagent(user_query, research_plan, st, tool_calls_per_subagent,
                     query_cache=query_cache,
                     document_collections=document_collections)
        for st in subtasks
    ]
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

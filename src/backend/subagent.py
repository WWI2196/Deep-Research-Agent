"""Subagent orchestration: search, evaluate, extract, report."""

import asyncio
import json
import logging
from typing import Any

from . import search as search_mod
from .helpers import enforce_source_diversity, extract_json, normalize_search_item
from .llm import chat
from .prompts import SOURCE_EVALUATE, SUBAGENT_REPORT

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

    # Deduplicate, limit 10
    seen: set[str] = set()
    deduped: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped.append(q)

    logger.info("Generated %d search queries from %d keywords for subtask %s",
                len(deduped[:10]), len(keywords), subtask.get("id", "?"))
    return deduped[:10]


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


async def run_subagent(
    user_query: str,
    research_plan: str,
    subtask: dict[str, Any],
    tool_budget: int,
) -> dict[str, Any]:
    sid = subtask["id"]
    stitle = subtask["title"]

    # 1 — generate queries (rules-based, no LLM call)
    queries = generate_search_queries(subtask)

    # 2 — parallel search
    search_queries = queries[: max(1, min(10, tool_budget))]

    async def _search_one(q: str) -> dict[str, Any] | None:
        try:
            return await asyncio.to_thread(search_mod.search, q, limit=8)
        except Exception:
            return None

    tasks = [_search_one(q) for q in search_queries]
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

    # 3b — adaptive query refinement
    refined_queries = await _refine_queries_if_needed(subtask, scored, queries)
    if refined_queries:
        refined_tasks = [_search_one(q) for q in refined_queries]
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
        if url in extracted_map:
            evidence.append({"url": url, "data": f"[FULL-TEXT] {extracted_map[url]}"})
        else:
            snippet = s.get("description") or s.get("title", "")
            if snippet:
                evidence.append({"url": url, "data": f"[SNIPPET] {snippet}"})

    evidence_text = "\n\n".join(f"[From {e['url']}]: {e['data']}" for e in evidence)

    # 6 — write report (retry once if too short)
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
                "content": f"Evidence:\n{evidence_text}" + (
                    "\n\nYOUR PREVIOUS RESPONSE WAS EMPTY OR TOO SHORT. Write a complete 800-1500 word report."
                    if attempt > 0 and len(report) < 200 else ""
                ),
            }],
        )
        if len(report) >= 200:
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
) -> dict[str, Any]:
    tasks = [
        run_subagent(user_query, research_plan, st, tool_calls_per_subagent)
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

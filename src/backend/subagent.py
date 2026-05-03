"""Subagent orchestration: search, evaluate, extract, report."""

import asyncio
import json
import logging
from typing import Any

from . import search as search_mod
from .helpers import enforce_source_diversity, extract_json, normalize_search_item
from .llm import chat
from .prompts import SOURCE_SCORING, SUBAGENT_REPORT, URL_SELECTION

logger = logging.getLogger(__name__)


async def generate_search_queries(subtask: dict[str, Any]) -> list[str]:
    prompt = (
        "Generate 4-7 diverse web search queries for the subtask below.\n"
        "Include broad, specific, natural-language, and entity-centric queries.\n"
        'Return JSON: {"queries": ["q1", "q2", ...]}\n\n'
        f"Subtask: {subtask['title']}\n"
        f"Description: {subtask['description']}\n"
        f"Objective: {subtask.get('objective', '')}\n"
        f"Preferred Sources: {subtask.get('source_types', '')}"
    )
    response = await chat(role="subagent", messages=[{"role": "user", "content": prompt}], temperature=0.3)
    try:
        payload = json.loads(extract_json(response))
        queries = payload.get("queries", [])
        source_types = subtask.get("source_types", "")
        if isinstance(source_types, str):
            source_types = [s.strip() for s in source_types.split(",")]

        modifiers: list[str] = []
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

        seen: set[str] = set()
        deduped: list[str] = []
        for q in final:
            if q not in seen:
                seen.add(q)
                deduped.append(q)
        return deduped[:10]
    except Exception:
        return [subtask["title"]]


async def batch_evaluate_sources(
    sources: list[dict[str, Any]],
    user_query: str,
) -> list[dict[str, Any]]:
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
                    {"role": "system", "content": SOURCE_SCORING.format(user_query=user_query)},
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
                src["reasoning"] = ev.get("reason", "") if ev else ""
                scored.append(src)
        except Exception:
            for src in batch:
                src["quality_score"] = 0.3
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

    # 1 — generate queries
    queries = await generate_search_queries(subtask)

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

    # 3 — evaluate
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
    sources_list = scored

    # Deduplicate
    seen_urls: set[str] = set()
    filtered: list[dict[str, Any]] = []
    for s in sources_list:
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

    top_urls = [s.get("url") for s in filtered if s.get("url")][: min(12, tool_budget)]
    if not top_urls:
        filtered = [{"url": rc.get("url"), "title": rc.get("title", ""),
                     "description": rc.get("description", ""), "quality_score": 0.2}
                    for rc in raw_candidates if rc.get("url") and rc.get("url") not in seen_urls]
        seen_urls.update(s.get("url") for s in filtered)
        top_urls = [s.get("url") for s in filtered if s.get("url")][: min(12, tool_budget)]

    # 4 — LLM selects sources worth deep-reading
    full_text_urls: set[str] = set()

    top_for_selection = [s for s in filtered if s.get("url") in top_urls][:12]
    if len(top_for_selection) > 2:
        sources_text = ""
        for i, s in enumerate(top_for_selection):
            desc = (s.get("description") or "")[:250]
            sources_text += (
                f"[{i}] score={s.get('quality_score', 0):.1f} | {s.get('title', '')[:120]}\n"
                f"    {desc}\n\n"
            )

        try:
            response = await chat(
                role="subagent",
                messages=[
                    {"role": "system", "content": URL_SELECTION.format(subtask_title=stitle)},
                    {"role": "user", "content": f"Select sources worth full-text reading:\n\n{sources_text}"},
                ],
                temperature=0.1,
            )
            indices = json.loads(extract_json(response)).get("indices", [])
            full_text_urls = {top_for_selection[i]["url"] for i in indices if i < len(top_for_selection)}
        except Exception as exc:
            logger.warning("URL selection failed, using top-5 by score: %s", exc)
            full_text_urls = {s["url"] for s in top_for_selection[:5] if s.get("url")}
    elif top_for_selection:
        full_text_urls = {s["url"] for s in top_for_selection if s.get("url")}

    # 5 — extract full text (markdown) via trafilatura for selected URLs
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

    # 6 — build evidence: full-text for extracted, snippet for the rest
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

    # 7 — write report
    report = await chat(
        role="subagent",
        messages=[{
            "role": "system",
            "content": SUBAGENT_REPORT.format(
                user_query=user_query,
                research_plan=research_plan,
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
            "content": f"Evidence:\n{evidence_text}",
        }],
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

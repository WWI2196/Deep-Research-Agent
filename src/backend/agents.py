"""Agent functions for each pipeline stage.

Every agent calls _chat() which routes through the configured async provider.
"""

import asyncio
import json
import logging
import re
from collections import Counter
from typing import Any
from urllib.parse import urlparse

from .config import get_config
from .prompts import (
    CITATION,
    PLANNER,
    REFLECTION,
    SCALING,
    SOURCE_SCORING,
    SPLITTER,
    SUBAGENT_REPORT,
    SYNTHESIS,
    URL_SELECTION,
)
from .providers import get_provider
from . import search as search_mod

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


def _pick_first_nonempty(item: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _normalize_search_item(item: dict[str, Any], source_label: str) -> dict[str, Any] | None:
    url = _pick_first_nonempty(
        item, ["url", "link", "sourceURL", "source_url", "href", "website", "canonical_url"]
    )
    if not url:
        return None
    title = _pick_first_nonempty(item, ["title", "name", "headline"]) or url
    description = _pick_first_nonempty(
        item, ["description", "snippet", "summary", "content", "markdown", "text"]
    )
    return {"title": title, "url": url, "description": description, "source": source_label}


def _has_clean_ending(text: str) -> bool:
    tail = text.rstrip()
    if not tail:
        return True
    if tail.endswith(("```", "***", "---", "___", "**")):
        return True
    return tail.endswith((".", "!", "?", ":", ")", "]", '"', "”", "'", "’"))


def _needs_continuation(text: str, end_marker: str | None = None) -> bool:
    if end_marker and end_marker not in text:
        return True
    if len(text) < 500:
        return False
    tail = text.rstrip()
    if _has_clean_ending(tail):
        return False
    if tail and tail[-1].isalnum():
        return True
    last_word_match = re.search(r"([A-Za-z]+)\W*$", tail)
    last_word = last_word_match.group(1).lower() if last_word_match else ""
    dangling = {"and", "the", "of", "in", "to", "a", "an", "or", "but", "for", "with", "that", "is", "are", "was", "were", "as"}
    return last_word in dangling


async def _continue_if_truncated(
    report: str,
    user_query: str,
    *,
    end_marker: str | None = None,
    max_rounds: int = 4,
) -> str:
    if not report:
        return report

    tail_chars = 3000
    for round_idx in range(max_rounds):
        if not _needs_continuation(report, end_marker):
            break

        logger.info("Report truncated, continuing (round %s/%s)", round_idx + 1, max_rounds)
        marker_instruction = f" End with the exact marker {end_marker}." if end_marker else ""
        try:
            continuation = await _chat(
                role="coordinator",
                messages=[{
                    "role": "system",
                    "content": (
                        "Continue this research report from where it was cut off. "
                        "Pick up EXACTLY where the text ends — do not repeat content. "
                        "Do not add preamble. Continue seamlessly."
                        f"{marker_instruction}"
                        f"The report is about: {user_query}"
                    ),
                }, {
                    "role": "user",
                    "content": f"Continue from where this was cut off:\n\n...{report[-tail_chars:]}",
                }],
                max_tokens=8192,
            )
            if not continuation or len(continuation.strip()) < 20:
                break
            report = report.rstrip() + "\n\n" + continuation.strip()
        except Exception as e:
            lower = str(e).lower()
            if any(t in lower for t in ["400", "too long", "max", "context"]):
                tail_chars = max(1000, int(tail_chars * 0.65))
                continue
            break

    if end_marker:
        report = report.replace(end_marker, "").rstrip()
    return report


# ── provider cache ───────────────────────────────────────────────────

_provider_cache: dict[str, object] = {}


def _get_or_create_provider(provider_name: str):
    if provider_name in _provider_cache:
        return _provider_cache[provider_name]

    app_cfg = get_config()
    pc = app_cfg.providers.get(provider_name)
    if not pc:
        # fallback to first available
        pc = app_cfg.providers.get(app_cfg.default_provider)
    if not pc:
        raise RuntimeError(f"No provider configured: '{provider_name}'")

    p = get_provider(pc.type, pc.base_url, pc.api_key)
    _provider_cache[provider_name] = p
    return p


# ── unified async LLM call ────────────────────────────────────────────

async def _chat(
    role: str,
    messages: list[dict[str, str]],
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_retries: int = 3,
) -> str:
    role_cfg = get_config().get_role(role)
    provider = _get_or_create_provider(role_cfg.provider)
    temp = temperature if temperature is not None else role_cfg.temperature

    for attempt in range(max_retries):
        try:
            result = await provider.chat(
                model=role_cfg.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens,
            )
            result = _clean_think_tags(result)
            return result
        except Exception as exc:
            err = str(exc).lower()
            if any(c in err for c in ["401", "403", "invalid"]):
                raise
            if attempt < max_retries - 1:
                delay = 2.0 * (2**attempt)
                logger.warning("LLM call [%s] attempt %d failed: %s. Retrying in %.1fs", role, attempt + 1, exc, delay)
                await asyncio.sleep(delay)
                continue
            raise

    raise RuntimeError(f"LLM call exhausted retries for [{role}]")


# ── pipeline agents ───────────────────────────────────────────────────

async def generate_research_plan(user_query: str) -> str:
    return await _chat(
        role="planner",
        messages=[
            {"role": "system", "content": PLANNER},
            {"role": "user", "content": user_query},
        ],
    )


async def split_into_subtasks(research_plan: str) -> list[dict[str, Any]]:
    response = await _chat(
        role="splitter",
        messages=[
            {"role": "system", "content": SPLITTER},
            {"role": "user", "content": research_plan + "\n\nReturn valid JSON."},
        ],
    )
    content = response.strip()
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        extracted = _extract_json(content)
        if not extracted:
            raise ValueError("Empty or invalid JSON from task splitter.")
        payload = json.loads(extracted)
    return payload["subtasks"]


async def compute_scaling(user_query: str, research_plan: str) -> dict[str, Any]:
    response = await _chat(
        role="scaler",
        messages=[{
            "role": "system",
            "content": SCALING,
        }, {
            "role": "user",
            "content": f"Query: {user_query}\n\nPlan:\n{research_plan}\n\nReturn valid JSON.",
        }],
        temperature=0.1,
    )
    content = response.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        extracted = _extract_json(content)
        if not extracted:
            raise ValueError("Empty response from scaler.")
        return json.loads(extracted)


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
    response = await _chat(role="subagent", messages=[{"role": "user", "content": prompt}], temperature=0.3)
    try:
        payload = json.loads(_extract_json(response))
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
            response = await _chat(
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
                result = json.loads(_extract_json(content))
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


def _enforce_source_diversity(
    sources: list[dict[str, Any]],
    max_per_domain: int = 3,
) -> list[dict[str, Any]]:
    domain_count: Counter = Counter()
    diverse: list[dict[str, Any]] = []
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
        response = await _chat(role="subagent", messages=[{"role": "user", "content": prompt}], temperature=0.4)
        payload = json.loads(_extract_json(response))
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
            normalized = _normalize_search_item(item, "search")
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
                normalized = _normalize_search_item(item, "refined-search")
                if normalized:
                    raw_candidates.append(normalized)
        scored = await batch_evaluate_sources(raw_candidates, subtask.get("objective", user_query))
        scored.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

    # 3c — enforce source diversity
    scored = _enforce_source_diversity(scored, max_per_domain=3)
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
            response = await _chat(
                role="subagent",
                messages=[
                    {"role": "system", "content": URL_SELECTION.format(subtask_title=stitle)},
                    {"role": "user", "content": f"Select sources worth full-text reading:\n\n{sources_text}"},
                ],
                temperature=0.1,
            )
            indices = json.loads(_extract_json(response)).get("indices", [])
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
    report = await _chat(
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


async def synthesize_report(
    user_query: str,
    research_plan: str,
    reports: list[str],
) -> str:
    combined = "\n\n".join(reports)
    max_chars = 80000

    for attempt in range(3):
        try:
            truncated = combined[:max_chars] if len(combined) > max_chars else combined
            result = await _chat(
                role="coordinator",
                messages=[{
                    "role": "system",
                    "content": SYNTHESIS.format(
                        user_query=user_query,
                        research_plan=research_plan[:3000],
                        subagent_reports=truncated,
                    ) + "\n\nIMPORTANT: End your full final answer with this exact marker on its own line: <<END_OF_REPORT>>",
                }],
                max_tokens=8192,
            )
            result = await _continue_if_truncated(result, user_query, end_marker="<<END_OF_REPORT>>", max_rounds=5)
            result = re.sub(r'\[([a-z0-9_]+(?:_[a-z0-9_]+)+)\]\s*', '', result)
            return result
        except Exception as e:
            lower = str(e).lower()
            if "400" in lower or "too long" in lower or "max" in lower:
                max_chars = int(max_chars * 0.6)
                continue
            raise

    return f"# Research Report: {user_query}\n\n{combined[:max_chars]}"


async def add_citations(report: str, sources: list[dict[str, Any]]) -> str:
    report = re.sub(r'\[([a-z0-9_]+(?:_[a-z0-9_]+)+)\]\s*', '', report)

    def _format_sources(limit: int, desc_limit: int) -> str:
        lines: list[str] = []
        seen_urls: set[str] = set()
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

    for report_limit, source_limit, desc_limit, out_tokens in attempt_plan:
        try:
            truncated_report = report[:report_limit] if len(report) > report_limit else report
            sources_text = _format_sources(source_limit, desc_limit)
            if not sources_text:
                return report

            result = await _chat(
                role="citation",
                messages=[{
                    "role": "system",
                    "content": CITATION.format(report=truncated_report, sources=sources_text),
                }],
                temperature=0.1,
                max_tokens=out_tokens,
            )
            result = await _continue_if_truncated(result, "citation pass", max_rounds=3)
            result = re.sub(r'\[([a-z0-9_]+(?:_[a-z0-9_]+)+)\]\s*', '', result)
            return result
        except Exception as e:
            lower = str(e).lower()
            if any(t in lower for t in ["400", "too long", "max", "context"]):
                continue
            break

    logger.warning("Citation agent failed after retries; returning uncited report.")
    return report

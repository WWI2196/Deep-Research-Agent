"""Synthesis and citation pipeline stages."""

import asyncio
import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

from . import search as search_mod
from .helpers import needs_continuation
from .llm import chat
from .prompts import FAILURE_SUMMARY, SYNTHESIS

logger = logging.getLogger(__name__)


async def _continue_if_truncated(
    report: str,
    user_query: str,
    *,
    end_marker: str = "<<END_OF_REPORT>>",
    max_rounds: int = 6,
) -> str:
    """Continue a truncated synthesis report using explicit continuation prompts.

    Unlike _continue_if_truncated which only sends the tail, this re-sends the
    original synthesis instruction plus the last segment of the report, asking
    the model to pick up exactly where it stopped.
    """
    if not report or (end_marker and end_marker in report):
        return report

    tail_chars = 4000
    for round_idx in range(max_rounds):
        if end_marker and end_marker in report:
            break
        if not needs_continuation(report, end_marker):
            break

        logger.info("Report truncated, continuing (round %s/%s)", round_idx + 1, max_rounds)
        tail = report[-tail_chars:]

        try:
            continuation = await chat(
                role="coordinator",
                messages=[{
                    "role": "system",
                    "content": (
                        "You are continuing a research report that was cut off mid-sentence. "
                        "Below is the LAST segment of the partial report. "
                        "Continue writing EXACTLY from where it stopped — do not add a "
                        "title, do not repeat any content, do not add preamble. "
                        "Write ONLY the continuation text seamlessly.\n\n"
                        f"The report topic is: {user_query}\n"
                        f"When you reach the end, write: {end_marker}"
                    ),
                }, {
                    "role": "user",
                    "content": f"Continue this incomplete report from its cutoff point:\n\n...{tail}",
                }],
                max_tokens=4096,
            )
            if not continuation or len(continuation.strip()) < 30:
                # Try with longer tail context
                tail_chars = min(8000, int(tail_chars * 1.5))
                if tail_chars > len(report):
                    break
                continue

            report = report.rstrip() + "\n\n" + continuation.strip()
            tail_chars = 4000  # reset for next round
        except Exception as e:
            lower = str(e).lower()
            if any(t in lower for t in ["400", "too long", "max", "context"]):
                tail_chars = max(2000, int(tail_chars * 0.6))
                continue
            break

    return report.replace(end_marker, "").rstrip() if end_marker else report


async def _generate_failure_summary(
    user_query: str,
    research_plan: str,
    reports: list[str],
    partial_report: str,
    reason: str,
    chat_fn=None,
) -> str:
    """Generate a compact failure summary for synthesis retry."""
    if chat_fn is None:
        chat_fn = chat

    report_tail = partial_report[-3000:] if partial_report else "(synthesis not yet attempted)"
    plan_tail = research_plan[:2000]

    try:
        response = await chat_fn(
            role="coordinator",
            messages=[{
                "role": "user",
                "content": FAILURE_SUMMARY.format(
                    user_query=user_query,
                    research_plan=plan_tail,
                    reports_count=len(reports),
                    partial_report=report_tail,
                    reason=reason,
                ),
            }],
            temperature=0.3,
            max_tokens=800,
        )
        summary = response.strip()
        return summary[:1000]
    except Exception as e:
        logger.warning("Failure summary generation failed: %s", e)
        return (
            f"The previous synthesis attempt was {reason}. "
            "Focus on completing all remaining sections and producing a full report."
        )


async def synthesize_report(
    user_query: str,
    research_plan: str,
    reports: list[str],
    failure_summary: str = "",
) -> str:
    """Multi-pass LLM synthesis with explicit continuation on truncation.

    Strategy:
    1. Generate main report with high max_tokens (16K)
    2. If output lacks <<END_OF_REPORT>>, explicitly continue in sequential rounds
    3. Fall back to concatenation only if synthesis completely fails
    """
    valid_reports = [r for r in reports if r and len(r.strip()) > 100]
    if not valid_reports:
        return f"# Research Report: {user_query}\n\n## Introduction\n\nNo research findings were collected.\n"

    try:
        plan = json.loads(research_plan)
    except (json.JSONDecodeError, TypeError):
        plan = {}
    methodology = plan.get("methodology", "")
    output_structure = json.dumps(plan.get("output_structure", []))

    combined = "\n\n---\n\n".join(valid_reports)
    max_input_chars = 60000
    report_input = combined[:max_input_chars]

    final_report = ""
    for attempt in range(3):
        try:
            # Main synthesis call — high max_tokens for a comprehensive report
            failure_block = (
                f"\n\nNote from previous attempt:\n{failure_summary}"
                if failure_summary else ""
            )

            result = await chat(
                role="coordinator",
                messages=[{
                    "role": "system",
                    "content": SYNTHESIS.format(
                        user_query=user_query,
                        methodology=methodology,
                        output_structure=output_structure,
                        subagent_reports=report_input,
                        failure_summary=failure_block,
                    ),
                }],
                max_tokens=16384,
            )

            # If truncated, explicitly continue until marker appears or no more truncation
            result = await _continue_if_truncated(
                result, user_query, end_marker="<<END_OF_REPORT>>", max_rounds=6,
            )

            # Remove any stray end markers
            result = result.replace("<<END_OF_REPORT>>", "").strip()

            if len(result) > 1000:
                final_report = result
                break

            logger.warning("Synthesis attempt %d too short (%d chars), retrying...",
                         attempt + 1, len(result))
            max_input_chars = int(max_input_chars * 0.7)
            report_input = combined[:max_input_chars]

        except Exception as e:
            lower = str(e).lower()
            if "400" in lower or "too long" in lower or "max" in lower:
                max_input_chars = int(max_input_chars * 0.6)
                report_input = combined[:max_input_chars]
                logger.warning("Synthesis attempt %d context error, reducing to %d chars",
                             attempt + 1, max_input_chars)
                continue
            logger.warning("Synthesis attempt %d failed: %s", attempt + 1, e)

    if len(final_report) < 1000:
        logger.warning("All synthesis attempts failed or short, falling back to concatenation")
        cleaned = [re.sub(r'\[[a-z0-9]+_[a-z0-9_]+\]\s*', '', r) for r in valid_reports]
        body = "\n\n".join(cleaned)
        final_report = (
            f"# Research Report: {user_query}\n\n"
            f"## Introduction\n\n{methodology}\n\n"
            f"{body}\n\n"
            f"## Conclusions\n\nKey findings are presented in the sections above.\n"
        )

    return final_report


def _normalize_url(url: str) -> str:
    """Clean URL: strip trailing punctuation, anchors, and query fragment markers."""
    url = url.strip().rstrip(".,;:!?)]}")
    # Strip #:~:text= anchors that trafilatura can't handle
    cut = url.find("#:~:text=")
    if cut != -1:
        url = url[:cut]
    return url


def _domain_from_url(url: str) -> str:
    """Extract a short domain label from a URL for use as fallback title."""
    try:
        domain = urlparse(url).netloc.replace("www.", "")
        return domain or url[:60]
    except Exception:
        return url[:60]


async def _verify_citation_urls(urls: list[str]) -> dict[str, bool]:
    """Concurrently check which URLs are accessible via trafilatura.

    Returns {url: True/False} — True if content was successfully extracted.
    """
    if not urls:
        return {}

    semaphore = asyncio.Semaphore(8)
    results: dict[str, bool] = {}

    async def _check_one(url: str) -> tuple[str, bool]:
        async with semaphore:
            try:
                text = await asyncio.wait_for(
                    asyncio.to_thread(search_mod.extract, url),
                    timeout=10,
                )
                return url, text is not None
            except Exception:
                return url, False

    tasks = [_check_one(u) for u in urls]
    gathered = await asyncio.gather(*tasks)
    for url, ok in gathered:
        results[url] = ok

    accessible = sum(1 for v in results.values() if v)
    logger.info("Citation check: %d/%d URLs accessible", accessible, len(urls))
    return results


async def add_citations(
    report: str, sources: list[dict[str, Any]],
) -> tuple[str, dict[str, bool]]:
    """Rule-based citation: parse [src: <url>] markers, number them, generate References.

    Returns (cited_report, verification_map) where verification_map is {url: accessible_bool}.
    """
    report = re.sub(r'\[[a-z0-9]+_[a-z0-9_]+\]\s*', '', report)

    src_pattern = re.compile(r'\[src:\s*(https?://[^\]]+)\]')
    raw_matches = src_pattern.findall(report)

    if not raw_matches:
        logger.info("No [src: url] markers found; append source list")
        return _append_source_list(report, sources), {}

    # Normalize and deduplicate URLs
    seen: dict[str, int] = {}
    normalized_map: dict[str, str] = {}  # raw -> normalized
    for url in raw_matches:
        clean = _normalize_url(url)
        normalized_map[url] = clean
        if clean not in seen:
            seen[clean] = len(seen) + 1

    def _replace_src(match):
        raw = match.group(1).strip()
        clean = normalized_map.get(raw, _normalize_url(raw))
        idx = seen.get(clean)
        return f"[^{idx}]" if idx else match.group(0)

    cited_report = src_pattern.sub(_replace_src, report)

    # Build source lookup
    source_by_url: dict[str, dict[str, Any]] = {}
    for s in sources:
        u = _normalize_url(s.get("url", ""))
        if u and u not in source_by_url:
            source_by_url[u] = s

    # Build References section
    refs = "\n\n## References\n\n"
    for url, idx in sorted(seen.items(), key=lambda x: x[1]):
        src = source_by_url.get(url, {})
        title = src.get("title", "") or _domain_from_url(url)
        refs += f"[^{idx}]: [{title}]({url})\n"

    cited_report += refs

    # Verify URL accessibility concurrently
    all_urls = list(seen.keys())
    verification = await _verify_citation_urls(all_urls)

    # Mark unverified URLs in References
    unverified_count = sum(1 for v in verification.values() if not v)
    if unverified_count > 0:
        for url, ok in verification.items():
            if not ok and url in seen:
                idx = seen[url]
                tag = "[unverified]"
                cited_report = cited_report.replace(
                    f"[^{idx}]: [{source_by_url.get(url, {}).get('title', '') or _domain_from_url(url)}]({url})",
                    f"[^{idx}]: [{source_by_url.get(url, {}).get('title', '') or _domain_from_url(url)}]({url}) {tag}",
                )

    # Append verification footer
    total = len(all_urls)
    accessible = total - unverified_count
    footer = f"\n\n---\n*Citation check: {accessible}/{total} URLs accessible*"
    cited_report += footer

    return cited_report, verification


def _append_source_list(report: str, sources: list[dict[str, Any]]) -> str:
    if not sources:
        return report
    seen_urls: set[str] = set()
    refs = "\n\n## Sources\n\n"
    idx = 1
    for s in sources:
        url = str(s.get("url", "")).strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        title = s.get("title", "Source")
        refs += f"{idx}. [{title}]({url})\n"
        idx += 1
    return report + refs

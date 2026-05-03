"""Synthesis and citation pipeline stages."""

import logging
import re
from typing import Any

from .helpers import needs_continuation
from .llm import chat
from .prompts import CITATION, SYNTHESIS

logger = logging.getLogger(__name__)


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
        if not needs_continuation(report, end_marker):
            break

        logger.info("Report truncated, continuing (round %s/%s)", round_idx + 1, max_rounds)
        marker_instruction = f" End with the exact marker {end_marker}." if end_marker else ""
        try:
            continuation = await chat(
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
            result = await chat(
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

            result = await chat(
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

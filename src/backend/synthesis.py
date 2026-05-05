"""Synthesis and citation pipeline stages."""

import json
import logging
import re
from typing import Any

from .helpers import needs_continuation
from .llm import chat
from .prompts import SYNTHESIS

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


async def synthesize_report(
    user_query: str,
    research_plan: str,
    reports: list[str],
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
            result = await chat(
                role="coordinator",
                messages=[{
                    "role": "system",
                    "content": SYNTHESIS.format(
                        user_query=user_query,
                        methodology=methodology,
                        output_structure=output_structure,
                        subagent_reports=report_input,
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


async def add_citations(report: str, sources: list[dict[str, Any]]) -> str:
    """Rule-based citation: parse [src: <url>] markers, number them, generate References."""
    report = re.sub(r'\[[a-z0-9]+_[a-z0-9_]+\]\s*', '', report)

    src_pattern = re.compile(r'\[src:\s*(https?://[^\]]+)\]')
    matches = src_pattern.findall(report)

    if not matches:
        logger.info("No [src: url] markers found; append source list")
        return _append_source_list(report, sources)

    seen: dict[str, int] = {}
    for url in matches:
        url = url.strip()
        if url not in seen:
            seen[url] = len(seen) + 1

    def _replace_src(match):
        url = match.group(1).strip()
        idx = seen.get(url)
        return f"[^{idx}]" if idx else match.group(0)

    cited_report = src_pattern.sub(_replace_src, report)

    refs = "\n\n## References\n\n"
    for url, idx in sorted(seen.items(), key=lambda x: x[1]):
        title = url
        for s in sources:
            if s.get("url", "").strip() == url:
                title = s.get("title", url)
                break
        refs += f"[^{idx}]: [{title}]({url})\n"

    return cited_report + refs


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

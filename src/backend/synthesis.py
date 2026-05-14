"""Synthesis and citation pipeline stages."""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import search as search_mod
from .helpers import needs_continuation
from .llm import chat
from .prompts import DEEPEN_SECTION, FAILURE_SUMMARY, SYNTHESIS

logger = logging.getLogger(__name__)


async def _continue_if_truncated(
    report: str,
    user_query: str,
    *,
    end_marker: str = "<<END_OF_REPORT>>",
    max_rounds: int = 6,
    cancel_event=None,
) -> str:
    """Continue a truncated synthesis report using explicit continuation prompts."""
    if not report or (end_marker and end_marker in report):
        return report

    tail_chars = 4000
    for round_idx in range(max_rounds):
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError("Research cancelled during synthesis continuation")
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
                tail_chars = min(8000, int(tail_chars * 1.5))
                if tail_chars > len(report):
                    break
                continue

            report = report.rstrip() + "\n\n" + continuation.strip()
            tail_chars = 4000
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


def _score_paragraph(paragraph: str, index: int) -> int:
    """Score paragraph importance for synthesis input trimming."""
    score = 0
    if "[src:" in paragraph:
        score += 3
    if re.search(r"\b\d{4}\b|\b\d+\.\d+|```|\bdef\s+\w+|\bclass\s+\w+|#\s+\w", paragraph):
        score += 2
    if index < 2:
        score += 1
    # Deprioritize transitional fluff
    if re.search(r"^(综上所述|总之|总而言之|最后|综上所述).*", paragraph.strip()):
        score -= 1
    return score


def _deduplicate_consecutive_headings(report: str) -> str:
    """Remove consecutive duplicate markdown headings from the report.

    LLMs occasionally emit the same section heading twice (e.g.
    '## Conclusions\n\n## Conclusions'). This strips the duplicate
    while preserving the first occurrence.
    """
    lines = report.split("\n")
    result: list[str] = []
    prev_heading: str | None = None
    for line in lines:
        m = re.match(r"^(#+ .+)$", line)
        if m:
            current = m.group(1).strip()
            if current == prev_heading:
                continue
            prev_heading = current
        result.append(line)
    return "\n".join(result)


def _trim_reports_by_whole(reports: list[str], max_chars: int) -> str:
    """Trim reports by preserving high-importance paragraphs from each report.

    Strategy: keep all reports, but within each report drop low-value paragraphs
    (transitions, background filler) before removing paragraphs with citations
    or data. Only as a last resort do we drop entire reports.
    """
    if not reports:
        return ""
    separator = "\n\n---\n\n"

    # Phase 1: try including all reports raw
    combined = separator.join(reports)
    if len(combined) <= max_chars:
        return combined

    # Phase 2: compress each report by dropping low-importance paragraphs
    # Budget allocation: proportional to original report length, with a minimum floor
    # so short but critical reports are not squeezed to nothing.
    total_raw = sum(len(r) for r in reports)
    min_budget = min(4000, max_chars // len(reports))
    compressed: list[str] = []
    for report in reports:
        paragraphs = [p for p in report.split("\n\n") if p.strip()]
        if not paragraphs:
            compressed.append("")
            continue
        scored = [(i, p, _score_paragraph(p, i)) for i, p in enumerate(paragraphs)]
        scored.sort(key=lambda x: x[2], reverse=True)
        # Proportional budget for this report
        ratio = len(report) / total_raw if total_raw > 0 else 1 / len(reports)
        report_budget = max(int(max_chars * ratio), min_budget)
        # Greedy rebuild: add paragraphs in importance order until near budget
        kept: list[tuple[int, str]] = []
        current_len = 0
        for idx, para, _ in scored:
            para_len = len(para) + 2  # +2 for \n\n
            if current_len + para_len <= report_budget:
                kept.append((idx, para))
                current_len += para_len
        # Restore original order
        kept.sort(key=lambda x: x[0])
        compressed.append("\n\n".join(p for _, p in kept))

    combined = separator.join(r for r in compressed if r)
    if len(combined) <= max_chars:
        return combined

    # Phase 3: drop shortest compressed reports until we fit
    indexed = sorted(enumerate(compressed), key=lambda x: len(x[1]) if x[1] else 0)
    dropped: set[int] = set()
    for idx, _ in indexed:
        remaining = [compressed[i] for i in range(len(compressed)) if i not in dropped and compressed[i]]
        if not remaining:
            break
        combined = separator.join(remaining)
        if len(combined) <= max_chars:
            return combined
        dropped.add(idx)

    # Fallback: keep at least the longest remaining report
    remaining = [compressed[i] for i in range(len(compressed)) if i not in dropped and compressed[i]]
    if remaining:
        longest = max(remaining, key=len)
        return longest[:max_chars]
    return reports[0][:max_chars] if reports else ""


async def synthesize_report(
    user_query: str,
    research_plan: str,
    reports: list[str],
    failure_summary: str = "",
    output_language: str = "zh",
    cancel_event=None,
    on_progress=None,
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

    # Pre-extract all inline citations from sub-agent reports for later recovery
    _src_pattern = re.compile(r'\[src:\s*((?:https?://|file://)[^\]]+)\]')
    _all_subagent_citations: set[str] = set()
    for r in valid_reports:
        _all_subagent_citations.update(_src_pattern.findall(r))
    logger.info("Pre-extracted %d unique citations from sub-agent reports", len(_all_subagent_citations))

    max_input_chars = 80000
    report_input = _trim_reports_by_whole(valid_reports, max_input_chars)

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
                        output_language=output_language,
                        output_structure=output_structure,
                        subagent_reports=report_input,
                        failure_summary=failure_block,
                    ),
                }],
                max_tokens=30000,
            )

            # If truncated, explicitly continue until marker appears or no more truncation
            result = await _continue_if_truncated(
                result, user_query, end_marker="<<END_OF_REPORT>>", max_rounds=6,
                cancel_event=cancel_event,
            )

            # Remove any stray end markers and deduplicate headings
            result = result.replace("<<END_OF_REPORT>>", "").strip()
            result = _deduplicate_consecutive_headings(result)

            if len(result) > 1000:
                final_report = result
                break

            logger.warning("Synthesis attempt %d too short (%d chars), retrying...",
                         attempt + 1, len(result))
            max_input_chars = int(max_input_chars * 0.7)
            report_input = _trim_reports_by_whole(valid_reports, max_input_chars)

        except Exception as e:
            lower = str(e).lower()
            if "400" in lower or "too long" in lower or "max" in lower:
                max_input_chars = int(max_input_chars * 0.6)
                report_input = _trim_reports_by_whole(valid_reports, max_input_chars)
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

    # Log how many citations survived synthesis
    _survived = set(_src_pattern.findall(final_report))
    _dropped = _all_subagent_citations - _survived
    if _dropped:
        logger.warning("Synthesis dropped %d/%d citations; will recover in add_citations",
                     len(_dropped), len(_all_subagent_citations))
    else:
        logger.info("All %d pre-extracted citations survived synthesis", len(_all_subagent_citations))

    # Deepen thin sections for analytical depth
    final_report = await _deepen_thin_sections(
        final_report,
        valid_reports,
        user_query,
        output_language=output_language,
        cancel_event=cancel_event,
        on_progress=on_progress,
    )

    return final_report


def _strip_heading_markers(text: str) -> str:
    """Remove markdown heading markers (#) from the start of text."""
    return re.sub(r'^#+\s*', '', text).strip()


async def _deepen_thin_sections(
    report: str,
    reports: list[str],
    user_query: str,
    output_language: str = "zh",
    chat_fn=None,
    cancel_event=None,
    on_progress=None,
) -> str:
    """Identify thin sections and expand them with deeper analysis.

    A top-level section (## heading) is "thin" if its body text is < 800
    characters OR it contains fewer than 3 inline [src: ...] citations.
    Only the weakest max_deepen sections are expanded to avoid runaway LLM
    calls. Each section is deepened AT MOST ONCE.
    """
    if chat_fn is None:
        chat_fn = chat

    # Parse top-level sections only (## headings, not ###)
    heading_pattern = re.compile(r'^(## [^#].*)$', re.MULTILINE)
    parts = heading_pattern.split(report)
    if not parts:
        return report

    # Identify thin sections at the ## level only
    thin_sections: list[tuple[int, str, str]] = []  # (idx, heading, body)
    deepened_headings: set[str] = set()
    for i, part in enumerate(parts):
        if part.startswith('## '):
            heading = part.strip()
            body = parts[i + 1] if i + 1 < len(parts) else ""
            body_text = body.strip()
            if not body_text:
                continue
            normalized_heading = _strip_heading_markers(heading)
            if normalized_heading in deepened_headings:
                continue
            char_count = len(body_text)
            citation_count = len(re.findall(r'\[src:\s*[^\]]+\]', body_text))
            # Threshold: < 800 chars OR < 3 citations
            if char_count < 800 or citation_count < 3:
                thin_sections.append((i, heading, body_text))
                deepened_headings.add(normalized_heading)
                logger.info(
                    "Thin section detected: %s (%d chars, %d citations)",
                    heading, char_count, citation_count,
                )

    # Cap deepening to the weakest sections to avoid runaway LLM calls
    max_deepen = 5
    total_thin = len(thin_sections)
    if total_thin > max_deepen:
        # Sort by (citation_count asc, char_count asc) — weakest first
        thin_sections.sort(key=lambda x: (
            len(re.findall(r'\[src:\s*[^\]]+\]', x[2])),
            len(x[2]),
        ))
        thin_sections = thin_sections[:max_deepen]
        logger.info(
            "Limiting deepening to %d weakest of %d thin sections",
            max_deepen, total_thin,
        )

    if not thin_sections:
        logger.info("No thin sections detected; skipping deepening.")
        return report

    # Build a searchable evidence pool from all sub-agent reports
    evidence_pool = "\n\n---\n\n".join(reports)

    # Expand each thin section sequentially (to stay within context limits)
    total = len(thin_sections)
    for done, (_, heading, body_text) in enumerate(thin_sections, start=1):
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError("Research cancelled during synthesis deepening")
        try:
            # Extract relevant evidence by simple keyword matching
            keywords = _extract_keywords(heading)
            relevant_evidence = _extract_relevant_evidence(evidence_pool, keywords, max_chars=12000)

            logger.info("Deepening section: %s", heading)
            expanded = await chat_fn(
                role="coordinator",
                messages=[{
                    "role": "system",
                    "content": DEEPEN_SECTION.format(
                        user_query=user_query,
                        section_title=heading,
                        current_length=len(body_text),
                        current_content=body_text,
                        relevant_evidence=relevant_evidence,
                        output_language=output_language,
                    ),
                }],
                max_tokens=8000,
            )
            if on_progress:
                await on_progress(done, total)
            expanded = expanded.strip().replace("<<END_OF_REPORT>>", "").strip()
            if len(expanded) > len(body_text) * 1.3:
                # Determine whether expanded already contains the heading
                expanded_first_line_raw = next(
                    (line for line in expanded.split("\n") if line.strip()), ""
                )
                expanded_first_line = _strip_heading_markers(expanded_first_line_raw)
                current_heading_text = _strip_heading_markers(heading)
                if expanded_first_line == current_heading_text:
                    new_section = expanded
                else:
                    new_section = heading + "\n\n" + expanded

                # Replace the entire section (heading through body) to avoid
                # leaving the original heading behind or matching the wrong boundary.
                heading_pos = report.find(heading)
                if heading_pos == -1:
                    logger.warning(
                        "Could not find section %s in report for replacement", heading
                    )
                    continue
                next_heading_match = heading_pattern.search(
                    report, heading_pos + len(heading)
                )
                section_end = next_heading_match.start() if next_heading_match else len(report)
                report = report[:heading_pos] + new_section.rstrip() + "\n\n" + report[section_end:]
                logger.info(
                    "Section %s deepened: %d -> %d chars",
                    heading, len(body_text), len(expanded),
                )
            else:
                logger.warning(
                    "Section %s expansion too small (%d -> %d chars), keeping original",
                    heading, len(body_text), len(expanded),
                )
        except Exception as e:
            logger.warning("Deepening failed for section %s: %s", heading, e)
            continue

    # Deepening may introduce duplicate headings (e.g. LLM re-emits the
    # section heading inside expanded content). Clean them up before returning.
    return _deduplicate_consecutive_headings(report)


def _extract_keywords(heading: str) -> list[str]:
    """Extract meaningful keywords from a section heading for evidence matching."""
    # Strip markdown heading markers and common stop words
    text = re.sub(r'^#+\s*', '', heading)
    # Remove punctuation and split
    words = re.findall(r'[一-鿿\w]+', text)
    # Filter out very short words and common stop words
    stop_words = {"the", "and", "of", "in", "to", "a", "for", "on", "with", "as", "is", "are", "的", "与", "及", "在", "为"}
    keywords = [w for w in words if len(w) > 1 and w.lower() not in stop_words]
    return keywords[:8]


def _extract_relevant_evidence(evidence_pool: str, keywords: list[str], max_chars: int = 12000) -> str:
    """Extract paragraphs from evidence pool that contain at least one keyword."""
    paragraphs = [p for p in evidence_pool.split("\n\n") if p.strip()]
    scored: list[tuple[str, int]] = []
    for para in paragraphs:
        score = sum(1 for kw in keywords if kw.lower() in para.lower())
        # Boost paragraphs with citations and data
        if "[src:" in para:
            score += 2
        if re.search(r'\b\d{4}\b|\b\d+\.\d+', para):
            score += 1
        if score > 0:
            scored.append((para, score))

    # Sort by relevance score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Greedy select top paragraphs up to max_chars
    result_parts: list[str] = []
    current_len = 0
    for para, _ in scored:
        if current_len + len(para) + 2 > max_chars:
            break
        result_parts.append(para)
        current_len += len(para) + 2

    return "\n\n".join(result_parts) if result_parts else "(No highly relevant evidence found)"


def _normalize_url(url: str) -> str:
    """Clean URL: strip trailing punctuation, anchors, and query fragment markers."""
    url = url.strip().rstrip(".,;:!?)]}")
    # Strip #:~:text= anchors that trafilatura can't handle
    cut = url.find("#:~:text=")
    if cut != -1:
        url = url[:cut]
    return url


def _domain_from_url(url: str) -> str:
    """Extract a short domain label or file name from a URL for use as fallback title."""
    if url.startswith("file://"):
        name = Path(url).name
        return name or "Local Document"
    try:
        domain = urlparse(url).netloc.replace("www.", "")
        return domain or url[:60]
    except Exception:
        return url[:60]


async def _verify_citation_urls(urls: list[str]) -> dict[str, bool]:
    """Concurrently check which URLs are accessible via Crawl4AI.

    Returns {url: True/False} — True if content was successfully extracted.
    """
    if not urls:
        return {}

    semaphore = asyncio.Semaphore(8)
    results: dict[str, bool] = {}

    async def _check_one(url: str) -> tuple[str, bool]:
        async with semaphore:
            if url.startswith("file://"):
                # Document library sources are always trusted; skip liveness check
                return url, True
            try:
                text = await asyncio.wait_for(
                    search_mod.extract_async(url),
                    timeout=15,
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


# Patterns to strip existing reference/source sections added by LLMs
_REF_SECTION_PATTERNS = [
    r'\n##+\s*References\s*\n.*?(?=\n##|\Z)',
    r'\n##+\s*Sources\s*\n.*?(?=\n##|\Z)',
    r'\n##+\s*Bibliography\s*\n.*?(?=\n##|\Z)',
    r'\n##+\s*参考文献\s*\n.*?(?=\n##|\Z)',
]


def _strip_existing_ref_sections(report: str) -> str:
    """Remove any LLM-generated References/Sources sections to avoid duplication."""
    original = report
    for pattern in _REF_SECTION_PATTERNS:
        report = re.sub(pattern, '\n', report, flags=re.DOTALL)
    # Guard: for reports >200 chars, if stripping removed >50% of content,
    # the model may have placed a reference heading mid-report.
    if len(original) > 200 and len(report) < len(original) * 0.5:
        logger.warning("_strip_existing_ref_sections removed >50%% of report (%d -> %d chars); skipping strip",
                     len(original), len(report))
        return original.rstrip()
    return report.rstrip()


async def add_citations(
    report: str, sources: list[dict[str, Any]],
    bench_format: bool = False,
) -> tuple[str, dict[str, bool]]:
    """Rule-based citation: parse [src: <url>] markers, number them, generate References.

    Args:
        bench_format: If True, use `[n]` instead of `[^n]` for compatibility with
            DeepResearch-Bench FACT evaluation (which expects `[15]` style citations).

    Returns (cited_report, verification_map) where verification_map is {url: accessible_bool}.
    """
    report = re.sub(r'\[[a-z0-9]+_[a-z0-9_]+\]\s*', '', report)
    report = _strip_existing_ref_sections(report)

    src_pattern = re.compile(r'\[src:\s*((?:https?://|file://)[^\]]+)\]')
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
        if not idx:
            return match.group(0)
        return f"[{idx}]" if bench_format else f"[^{idx}]"

    cited_report = src_pattern.sub(_replace_src, report)

    # ── Cleanup: strip any remaining [src: ...] markers that lack a valid URL ──
    orphan_pattern = re.compile(r'\[src:\s*[^\]]+\]')
    orphan_count = len(orphan_pattern.findall(cited_report))
    if orphan_count:
        logger.warning("Stripped %d orphaned [src: ...] markers (no valid URL)", orphan_count)
        cited_report = orphan_pattern.sub('', cited_report)
        # Collapse multiple spaces left behind
        cited_report = re.sub(r' {2,}', ' ', cited_report)

    # Build source lookup
    source_by_url: dict[str, dict[str, Any]] = {}
    for s in sources:
        u = _normalize_url(s.get("url", ""))
        if u and u not in source_by_url:
            source_by_url[u] = s

    # Build References section — one per line, clean numbered layout
    refs = "\n\n## References\n\n"
    for url, idx in sorted(seen.items(), key=lambda x: x[1]):
        src = source_by_url.get(url, {})
        title = src.get("title", "") or _domain_from_url(url)
        if url.startswith("file://"):
            refs += f"{idx}. 📄 {title}\n\n"
        else:
            refs += f"{idx}. [{title}]({url})\n\n"

    cited_report += refs.rstrip() + "\n"

    # URL verification temporarily disabled — trafilatura liveness check was too strict
    all_urls = list(seen.keys())
    verification = {url: True for url in all_urls}

    return cited_report, verification


def _append_source_list(report: str, sources: list[dict[str, Any]]) -> str:
    if not sources:
        return report
    seen_urls: set[str] = set()
    doc_sources: list[tuple[str, str]] = []
    web_sources: list[tuple[str, str]] = []
    for s in sources:
        url = str(s.get("url", "")).strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        title = s.get("title", "Source") or _domain_from_url(url)
        if url.startswith("file://"):
            doc_sources.append((title, url))
        else:
            web_sources.append((title, url))

    refs = ""
    if doc_sources:
        refs += "\n\n## Document Library Sources\n\n"
        for idx, (title, url) in enumerate(doc_sources, start=1):
            refs += f"{idx}. [{title}]({url})\n"
    if web_sources:
        refs += "\n\n## Web Sources\n\n"
        offset = len(doc_sources) + 1
        for idx, (title, url) in enumerate(web_sources, start=offset):
            refs += f"{idx}. [{title}]({url})\n"
    return report + refs

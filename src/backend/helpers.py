"""Pure text and JSON utility helpers — no LLM or external dependencies."""

import re
from collections import Counter, defaultdict
from typing import Any
from urllib.parse import urlparse


def extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return ""


def clean_think_tags(content: str) -> str:
    if "<think>" in content and "</think>" in content:
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


def pick_first_nonempty(item: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def normalize_search_item(item: dict[str, Any], source_label: str) -> dict[str, Any] | None:
    url = pick_first_nonempty(
        item, ["url", "link", "sourceURL", "source_url", "href", "website", "canonical_url"]
    )
    if not url:
        return None
    title = pick_first_nonempty(item, ["title", "name", "headline"]) or url
    description = pick_first_nonempty(
        item, ["description", "snippet", "summary", "content", "markdown", "text"]
    )
    return {"title": title, "url": url, "description": description, "source": item.get("source") or source_label}


def has_clean_ending(text: str) -> bool:
    tail = text.rstrip()
    if not tail:
        return True
    if tail.endswith(("```", "***", "---", "___", "**")):
        return True
    return tail.endswith((".", "!", "?", ":", ")", "]", '"', "”", "'", "’"))


def needs_continuation(text: str, end_marker: str | None = None) -> bool:
    if end_marker and end_marker not in text:
        return True
    if len(text) < 500:
        return False
    tail = text.rstrip()
    if has_clean_ending(tail):
        return False
    if tail and tail[-1].isalnum():
        return True
    last_word_match = re.search(r"([A-Za-z]+)\W*$", tail)
    last_word = last_word_match.group(1).lower() if last_word_match else ""
    dangling = {"and", "the", "of", "in", "to", "a", "an", "or", "but", "for", "with", "that", "is", "are", "was", "were", "as"}
    return last_word in dangling


def enforce_source_diversity(
    sources: list[dict[str, Any]],
    max_per_domain: int = 3,
) -> list[dict[str, Any]]:
    domain_count: Counter = Counter()
    diverse: list[dict[str, Any]] = []
    for s in sources:
        url = s.get("url", "")
        try:
            if url.startswith("file://"):
                from pathlib import Path
                domain = Path(url).parent.as_posix()
            else:
                domain = urlparse(url).netloc.replace("www.", "")
        except Exception:
            domain = url
        if domain_count[domain] < max_per_domain:
            domain_count[domain] += 1
            diverse.append(s)
    return diverse


def query_similarity(q1: str, q2: str) -> float:
    """Jaccard similarity between two query strings on word tokens."""
    words1 = set(q1.lower().split())
    words2 = set(q2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


def enforce_source_type_quota(
    sources: list[dict[str, Any]],
    quotas: dict[str, int] | None = None,
    min_per_type: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Enforce per-source-type quotas and minimums while preserving quality ordering.

    Default quotas: document=3, web=8. Default minimums: document=1, web=2.
    """
    quotas = quotas if isinstance(quotas, dict) else {"document": 3, "web": 8}
    min_per_type = min_per_type if isinstance(min_per_type, dict) else {"document": 1, "web": 2}

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in sources:
        stype = s.get("source", "web")
        by_type[stype].append(s)

    result: list[dict[str, Any]] = []
    taken: dict[str, int] = {k: 0 for k in quotas}
    taken["other"] = 0

    # Phase 1 — satisfy minimums per type
    for stype, minimum in min_per_type.items():
        items = by_type.get(stype, [])
        for item in items[:minimum]:
            result.append(item)
            taken[stype] = taken.get(stype, 0) + 1

    # Phase 2 — fill remaining slots up to quotas, then overflow into global pool
    overflow: list[dict[str, Any]] = []
    for stype, items in by_type.items():
        already = sum(1 for r in result if r.get("source", "web") == stype)
        quota = quotas.get(stype, len(items))
        for item in items[already:]:
            if taken.get(stype, 0) < quota:
                result.append(item)
                taken[stype] = taken.get(stype, 0) + 1
            else:
                overflow.append(item)

    # Phase 3 — append overflow items sorted by quality_score
    overflow.sort(key=lambda x: x.get("quality_score", 0.0), reverse=True)
    result.extend(overflow)
    return result


def smart_truncate(text: str, max_chars: int) -> str:
    """Truncate text at a natural boundary (section, paragraph, sentence).

    Prefers preserving whole sections (## headers) or paragraphs.
    Falls back to sentence boundary, then word boundary.
    """
    if len(text) <= max_chars:
        return text

    # Try section boundary (markdown ## headers)
    sections = re.split(r"\n## ", text)
    if len(sections) > 1:
        result = sections[0]
        for sec in sections[1:]:
            candidate = result + "\n## " + sec
            if len(candidate) > max_chars * 0.95:
                break
            result = candidate
        return result + "\n\n...[truncated]"

    # Try paragraph boundary
    truncate_at = text.rfind("\n\n", 0, max_chars - 20)
    if truncate_at == -1:
        truncate_at = text.rfind("\n", 0, max_chars - 20)

    # Try sentence boundary
    if truncate_at == -1:
        truncate_at = text.rfind(". ", 0, max_chars - 20)

    # Fallback to word boundary
    if truncate_at == -1:
        truncate_at = text.rfind(" ", 0, max_chars - 20)

    if truncate_at == -1:
        truncate_at = max_chars - 20

    return text[:truncate_at] + "\n\n...[truncated]"


def estimate_tokens(text: str) -> int:
    """Rough token estimate: chars / 3 (good enough for budgeting)."""
    return len(text) // 3


def generate_broader_queries(query: str) -> list[str]:
    """Generate up to 2 broader variant queries by stripping known modifiers."""
    modifiers = [
        "research paper", "study", "pdf", "official", "documentation",
        "github", "source code", "latest", "market report", "industry analysis",
        "statistics", "dataset", "analysis", "review", "guide",
    ]
    lower_query = query.lower()
    alternatives: list[str] = []
    for mod in modifiers:
        if mod in lower_query:
            alt = re.sub(re.escape(mod), "", query, flags=re.IGNORECASE).strip()
            alt = re.sub(r"\s+", " ", alt).strip()
            if alt and len(alt) > 3 and alt.lower() != lower_query:
                alternatives.append(alt)
            if len(alternatives) >= 2:
                break
    if not alternatives:
        words = query.split()
        if len(words) > 3:
            alt = " ".join(words[1:])
            if len(alt) > 3:
                alternatives.append(alt)
    return alternatives[:2]

"""Pure text and JSON utility helpers — no LLM or external dependencies."""

import re
from collections import Counter
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
    return {"title": title, "url": url, "description": description, "source": source_label}


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

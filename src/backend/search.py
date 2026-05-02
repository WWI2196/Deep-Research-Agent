"""Search layer — SearXNG for search, trafilatura for content extraction.

Always returns ``{"data": [...]}`` shape for search, plain markdown for extract.
"""

import logging
import os
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

SEARXNG_BASE = os.environ.get("SEARXNG_BASE", "http://127.0.0.1:8080")


# ── SearXNG search ──────────────────────────────────────────────

def _searxng_search(query: str, limit: int = 10) -> dict[str, Any]:
    """Search via self-hosted SearXNG JSON API (70+ engines aggregated)."""
    params = urlencode({"q": query, "format": "json", "categories": "general, science"})
    url = f"{SEARXNG_BASE}/search?{params}"
    try:
        req = Request(url, headers={"User-Agent": "DeepResearchAgent/1.0"})
        with urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as exc:
        logger.warning("SearXNG search failed: %s", exc)
        return {"data": []}

    import json
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("SearXNG returned invalid JSON")
        return {"data": []}

    results = payload.get("results", [])
    items: list[dict] = []
    for r in results[:limit]:
        items.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "description": r.get("content", "") or r.get("snippet", ""),
            "score": r.get("score", 0),
            "engines": r.get("engines", []),
            "category": r.get("category", ""),
            "source": "searxng",
        })
    return {"data": items}


# ── trafilatura content extraction ──────────────────────────────

def extract(url: str) -> str | None:
    """Fetch page content as clean markdown via trafilatura. Returns None on failure."""
    try:
        import trafilatura
    except ImportError:
        logger.warning("trafilatura not installed; extract disabled")
        return None

    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded, output_format="markdown", with_metadata=True)
        return text.strip() if text else None
    except Exception as exc:
        logger.warning("trafilatura extract failed for %s: %s", url[:60], exc)
        return None


# ── Public API ───────────────────────────────────────────────────

def search(query: str, limit: int = 10) -> dict[str, Any]:
    """Search via SearXNG. Falls back to empty result on error."""
    result = _searxng_search(query, limit=limit)
    if result.get("data"):
        return result
    return {"data": []}

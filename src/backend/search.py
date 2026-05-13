"""Search layer — SearXNG for search, Crawl4AI for content extraction.

Always returns ``{"data": [...]}`` shape for search, plain markdown for extract.
"""

import asyncio
import io
import logging
import os
import re
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


# ── PDF extraction ──────────────────────────────────────────────

def _extract_pdf(url: str) -> str | None:
    """Extract text from a PDF URL using pdfplumber."""
    try:
        import pdfplumber
        import requests
    except ImportError:
        logger.warning("pdfplumber not installed; PDF extraction disabled")
        return None

    try:
        resp = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"
                ),
                "Accept": "application/pdf,*/*;q=0.9",
            },
            timeout=20,
        )
        resp.raise_for_status()
        text_parts: list[str] = []
        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            for page in pdf.pages[:12]:  # limit to first 12 pages
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        full_text = "\n\n".join(text_parts).strip()
        return full_text if full_text else None
    except Exception as exc:
        logger.warning("PDF extraction failed for %s: %s", url[:60], exc)
        return None


def _resolve_url(url: str) -> str:
    """Resolve special URLs to extraction-friendly forms.

    - arXiv abs pages -> PDF direct link
    """
    arxiv_match = re.match(r"https?://arxiv\.org/abs/([\d\.]+)", url)
    if arxiv_match:
        return f"https://arxiv.org/pdf/{arxiv_match.group(1)}.pdf"
    return url


# ── Crawl4AI async extraction ───────────────────────────────────

async def extract_async(url: str, min_length: int = 500) -> str | None:
    """Fetch page content via Crawl4AI headless browser. Returns None on failure."""
    resolved = _resolve_url(url)

    # PDF path — run sync extractor in thread
    if resolved.lower().endswith(".pdf"):
        return await asyncio.to_thread(_extract_pdf, resolved)

    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

        run_conf = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

        async def _try_crawl(channel: str | None = None) -> str | None:
            kwargs: dict[str, Any] = {"headless": True, "verbose": False}
            if channel:
                kwargs["channel"] = channel
            browser_conf = BrowserConfig(**kwargs)
            async with AsyncWebCrawler(config=browser_conf) as crawler:
                result = await asyncio.wait_for(
                    crawler.arun(url=resolved, config=run_conf),
                    timeout=30,
                )
                if result and result.markdown:
                    md = result.markdown
                    if hasattr(md, "raw_markdown"):
                        md = md.raw_markdown
                    return md.strip() if md else None
            return None

        text = await _try_crawl()
        if text is None:
            # Retry with system Chrome if bundled Chromium is missing
            text = await _try_crawl(channel="chrome")
        if text and len(text.strip()) >= min_length:
            return text
        return text if text else None
    except ImportError:
        logger.warning("crawl4ai not installed; extract disabled")
        return None
    except asyncio.TimeoutError:
        logger.warning("Crawl4AI timeout for %s", resolved[:60])
        return None
    except Exception as exc:
        logger.warning("Crawl4AI extract failed for %s: %s", resolved[:60], exc)
        return None


# ── Public API ───────────────────────────────────────────────────

def search(query: str, limit: int = 10) -> dict[str, Any]:
    """Search via SearXNG. Falls back to empty result on error."""
    result = _searxng_search(query, limit=limit)
    if result.get("data"):
        return result
    return {"data": []}

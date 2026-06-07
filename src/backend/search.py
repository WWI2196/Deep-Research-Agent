"""Search layer — SearXNG for search, Crawl4AI for content extraction.

Always returns ``{"data": [...]}`` shape for search, plain markdown for extract.

Optimizations:
- SearXNG: async via aiohttp with persistent connection pool (no sync urlopen).
- Crawl4AI: singleton browser pool with asyncio.Semaphore for concurrency control.
  Browser is started once and reused across extractions, avoiding per-URL startup cost.
"""

import asyncio
import io
import json
import logging
import os
import re
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

SEARXNG_BASE = os.environ.get("SEARXNG_BASE", "http://127.0.0.1:8080")


# ── Shared aiohttp session (connection pool for SearXNG + general HTTP) ──

_http_session: aiohttp.ClientSession | None = None
_http_session_lock = asyncio.Lock()


async def _get_http_session() -> aiohttp.ClientSession:
    """Return a lazily-initialised aiohttp session with connection pooling."""
    global _http_session
    if _http_session is None or _http_session.closed:
        async with _http_session_lock:
            if _http_session is None or _http_session.closed:
                connector = aiohttp.TCPConnector(
                    limit=30,
                    limit_per_host=10,
                    ttl_dns_cache=300,
                    enable_cleanup_closed=True,
                )
                timeout = aiohttp.ClientTimeout(total=15)
                _http_session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers={"User-Agent": "DeepResearchAgent/1.0"},
                )
    return _http_session


# ── Crawl4AI browser pool ─────────────────────────────────────────

_crawler = None
_crawler_lock = asyncio.Lock()
_crawl_semaphore = asyncio.Semaphore(4)  # max concurrent browser extractions


async def _get_crawler():
    """Return a lazily-initialised shared Crawl4AI crawler (browser stays alive)."""
    global _crawler
    if _crawler is not None:
        return _crawler
    async with _crawler_lock:
        if _crawler is not None:
            return _crawler
        from crawl4ai import AsyncWebCrawler, BrowserConfig

        browser_conf = BrowserConfig(headless=True, verbose=False)
        crawler = AsyncWebCrawler(config=browser_conf)
        await crawler.__aenter__()
        _crawler = crawler
        logger.info("Crawl4AI browser pool initialised")
        return _crawler


async def _reset_crawler():
    """Destroy and re-create the browser on next call (e.g. after crash)."""
    global _crawler
    async with _crawler_lock:
        if _crawler is not None:
            try:
                await _crawler.__aexit__(None, None, None)
            except Exception:
                pass
            _crawler = None


async def cleanup():
    """Release all shared resources — call on server shutdown."""
    global _http_session, _crawler
    if _http_session is not None:
        try:
            await _http_session.close()
        except Exception:
            pass
        _http_session = None
    if _crawler is not None:
        try:
            await _crawler.__aexit__(None, None, None)
        except Exception:
            pass
        _crawler = None
    logger.info("search module resources cleaned up")


# ── SearXNG search (async) ──────────────────────────────────────

async def _searxng_search(query: str, limit: int = 10) -> dict[str, Any]:
    """Search via self-hosted SearXNG JSON API (70+ engines aggregated)."""
    params = {"q": query, "format": "json", "categories": "general, science"}
    try:
        session = await _get_http_session()
        async with session.get(
            f"{SEARXNG_BASE}/search", params=params,
        ) as resp:
            raw = await resp.text()
    except Exception as exc:
        logger.warning("SearXNG search failed: %s", exc)
        return {"data": []}

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

async def _extract_pdf(url: str) -> str | None:
    """Extract text from a PDF URL using pdfplumber + shared aiohttp pool."""
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed; PDF extraction disabled")
        return None

    try:
        session = await _get_http_session()
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            ),
            "Accept": "application/pdf,*/*;q=0.9",
        }
        async with session.get(url, headers=headers) as resp:
            resp.raise_for_status()
            content = await resp.read()
        text_parts: list[str] = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages[:12]:
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


# ── Crawl4AI async extraction (pool-based) ──────────────────────

async def extract_async(url: str, min_length: int = 500) -> str | None:
    """Fetch page content. Tries trafilatura first (fast), falls back to Crawl4AI browser.

    trafilatura handles most static HTML pages in ~0.2s.
    Crawl4AI (browser pool) is only used for JS-heavy pages that trafilatura can't handle.
    """
    resolved = _resolve_url(url)

    # PDF path — async extraction via shared aiohttp pool
    if resolved.lower().endswith(".pdf"):
        return await _extract_pdf(resolved)

    # Fast path: trafilatura (no browser needed, ~10-100x faster)
    text = await _extract_trafilatura(resolved)
    if text and len(text) >= min_length:
        return text

    # Slow path: Crawl4AI browser pool (for JS-rendered pages)
    try:
        from crawl4ai import CacheMode, CrawlerRunConfig
    except ImportError:
        logger.warning("crawl4ai not installed; extract disabled")
        return text if text else None

    run_conf = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    async with _crawl_semaphore:
        try:
            crawler = await _get_crawler()
            result = await asyncio.wait_for(
                crawler.arun(url=resolved, config=run_conf),
                timeout=15,
            )
            if result and result.markdown:
                md = result.markdown
                if hasattr(md, "raw_markdown"):
                    md = md.raw_markdown
                if md and len(md.strip()) >= min_length:
                    return md.strip()
                return md.strip() if md else (text if text else None)
            return text if text else None
        except TimeoutError:
            logger.warning("Crawl4AI pool timeout for %s", resolved[:60])
            fallback = await _extract_chrome_fallback(resolved, run_conf)
            return fallback or text
        except Exception as exc:
            logger.warning("Crawl4AI pool failed for %s: %s — resetting pool", resolved[:60], exc)
            await _reset_crawler()
            fallback = await _extract_chrome_fallback(resolved, run_conf)
            return fallback or text


async def _extract_trafilatura(url: str) -> str | None:
    """Fast extraction via trafilatura (no browser). Returns None on failure."""
    try:
        import trafilatura
    except ImportError:
        return None

    def _sync_extract() -> str | None:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        return trafilatura.extract(
            downloaded,
            include_links=True,
            include_tables=True,
            favor_recall=True,
        )

    try:
        return await asyncio.to_thread(_sync_extract)
    except Exception as exc:
        logger.debug("trafilatura failed for %s: %s", url[:60], exc)
        return None


async def _extract_chrome_fallback(url: str, run_conf) -> str | None:
    """Fallback: standalone Crawl4AI with system Chrome channel."""
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig

        browser_conf = BrowserConfig(headless=True, verbose=False, channel="chrome")
        async with AsyncWebCrawler(config=browser_conf) as crawler:
            result = await asyncio.wait_for(
                crawler.arun(url=url, config=run_conf),
                timeout=20,
            )
            if result and result.markdown:
                md = result.markdown
                if hasattr(md, "raw_markdown"):
                    md = md.raw_markdown
                return md.strip() if md else None
        return None
    except Exception:
        return None


# ── Public API ───────────────────────────────────────────────────

async def search(query: str, limit: int = 10) -> dict[str, Any]:
    """Search via SearXNG (async). Falls back to empty result on error."""
    result = await _searxng_search(query, limit=limit)
    if result.get("data"):
        return result
    return {"data": []}

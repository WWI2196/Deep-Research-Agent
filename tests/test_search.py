"""Tests for search layer — SearXNG search + trafilatura extract."""

import json
import os
import pytest
from unittest.mock import MagicMock, patch


# ── SearXNG search ──────────────────────────────────────────────


def test_searxng_search_success():
    from src.backend import search
    mock_response = json.dumps({
        "query": "test query",
        "number_of_results": 3,
        "results": [
            {"title": "Result 1", "url": "https://example.com/1", "content": "Description 1", "score": 5.0, "engines": ["google", "bing"]},
            {"title": "Result 2", "url": "https://example.com/2", "content": "Description 2", "score": 3.0, "engines": ["duckduckgo"]},
            {"title": "Result 3", "url": "https://example.com/3", "snippet": "Snippet 3", "score": 1.0, "engines": ["wikipedia"]},
        ]
    })

    with patch("src.backend.search.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value.read.return_value = mock_response.encode()
        result = search.search("test query", limit=5)
    assert len(result["data"]) == 3
    assert result["data"][0]["title"] == "Result 1"
    assert result["data"][0]["score"] == 5.0
    assert result["data"][0]["source"] == "searxng"


def test_searxng_search_network_error():
    from src.backend import search
    with patch("src.backend.search.urlopen", side_effect=OSError("Connection refused")):
        assert search.search("test") == {"data": []}


def test_searxng_search_invalid_json():
    from src.backend import search
    with patch("src.backend.search.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value.read.return_value = b"not json"
        assert search.search("test") == {"data": []}


def test_searxng_search_respects_limit():
    from src.backend import search
    results = [{"title": f"R{i}", "url": f"https://e.g/{i}", "content": f"D{i}", "score": 1.0, "engines": []} for i in range(15)]
    mock_response = json.dumps({"query": "test", "results": results})
    with patch("src.backend.search.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value.read.return_value = mock_response.encode()
        assert len(search.search("test", limit=5)["data"]) == 5


# ── trafilatura extract ────────────────────────────────────────


def test_extract_success():
    from src.backend import search
    mock_resp = MagicMock()
    mock_resp.text = "<html><body><article>Page content</article></body></html>"
    mock_resp.raise_for_status = MagicMock()
    with patch("requests.get", return_value=mock_resp), \
         patch("trafilatura.extract", return_value="Page content"):
        assert search.extract("https://example.com") == "Page content"


def test_extract_download_failed():
    from src.backend import search
    with patch("requests.get", side_effect=Exception("Network error")):
        assert search.extract("https://example.com") is None


def test_extract_import_error():
    from src.backend import search
    with patch.dict("sys.modules", {"trafilatura": None}):
        # trafilatura not installed
        pass
    with patch("builtins.__import__", side_effect=ImportError("no trafilatura")):
        assert search.extract("https://example.com") is None


# ── extract_async (trafilatura + Crawl4AI fallback) ─────────────


@pytest.mark.asyncio
async def test_extract_async_trafilatura_success():
    from src.backend import search
    mock_resp = MagicMock()
    mock_resp.text = "<html><body><article>Long page content here</article></body></html>"
    mock_resp.raise_for_status = MagicMock()
    with patch("requests.get", return_value=mock_resp), \
         patch("trafilatura.extract", return_value="Long page content here " * 50):
        result = await search.extract_async("https://example.com")
    assert result is not None
    assert len(result) >= 500


@pytest.mark.asyncio
async def test_extract_async_fallback_to_crawl4ai():
    from src.backend import search
    # Build a mock crawl4ai module so the lazy import inside extract_async works
    mock_crawl4ai = MagicMock()
    mock_crawler_cls = MagicMock()
    mock_crawler = mock_crawler_cls.return_value.__aenter__.return_value
    mock_result = mock_crawler.arun.return_value
    mock_result.markdown = "Crawl4AI rendered markdown content"
    mock_crawl4ai.AsyncWebCrawler = mock_crawler_cls
    mock_crawl4ai.BrowserConfig = MagicMock()
    mock_crawl4ai.CrawlerRunConfig = MagicMock()
    mock_crawl4ai.CacheMode = MagicMock()

    mock_resp = MagicMock()
    mock_resp.text = "<html><body>Short</body></html>"
    mock_resp.raise_for_status = MagicMock()

    # trafilatura returns short text — should trigger Crawl4AI fallback
    with patch("requests.get", return_value=mock_resp), \
         patch("trafilatura.extract", return_value="Short"):
        with patch.dict("sys.modules", {"crawl4ai": mock_crawl4ai}):
            result = await search.extract_async("https://example.com")
    assert result == "Crawl4AI rendered markdown content"


@pytest.mark.asyncio
async def test_extract_async_crawl4ai_not_installed():
    from src.backend import search
    mock_resp = MagicMock()
    mock_resp.text = "<html><body>Short</body></html>"
    mock_resp.raise_for_status = MagicMock()
    with patch("requests.get", return_value=mock_resp), \
         patch("trafilatura.extract", return_value="Short"):
        with patch.dict("sys.modules", {"crawl4ai": None}):
            result = await search.extract_async("https://example.com")
    # Falls back to trafilatura result even if short
    assert result == "Short"


# ── Real-world smoke test (previously failing URLs) ─────────────

# These URLs were extracted from production trace logs where trafilatura
# returned 0 succeeded.  They serve as a regression suite for Crawl4AI.
_PREVIOUSLY_FAILING_URLS = [
    "https://zhuanlan.zhihu.com/p/99037925",          # Zhihu column (JS + anti-bot)
    "https://blog.csdn.net/software444/article/details/151366444",  # CSDN blog
    "https://cloud.tencent.com/developer/article/2189831",          # Tencent Cloud
    "https://finance.sina.com.cn/stock/zqgd/2025-04-24/doc-ineufhcf3801544.shtml",  # Sina Finance
]


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_TESTS") != "1",
    reason="Set RUN_SLOW_TESTS=1 to run live network smoke tests",
)
@pytest.mark.asyncio
async def test_extract_async_previously_failing_urls():
    from src.backend.search import extract_async

    results: dict[str, str | None] = {}
    for url in _PREVIOUSLY_FAILING_URLS:
        try:
            text = await extract_async(url, min_length=300)
            results[url] = text
        except Exception as exc:
            results[url] = f"EXCEPTION: {exc}"

    successes = [
        url for url, text in results.items()
        if text and len(text) >= 300 and not text.startswith("EXCEPTION")
    ]

    # Print diagnostic info so pytest -s shows what happened
    for url, text in results.items():
        status = "OK" if url in successes else "FAIL"
        length = len(text) if text else 0
        print(f"[{status}] {length:>5} chars  {url}")

    # We expect Crawl4AI to recover at least half of these URLs.
    # Individual sites may still block us, so the bar is intentionally modest.
    assert len(successes) >= 2, (
        f"Expected at least 2 recoveries out of {len(_PREVIOUSLY_FAILING_URLS)}, "
        f"got {len(successes)}. Full results:\n{results}"
    )

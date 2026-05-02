"""Tests for search layer — SearXNG search + trafilatura extract."""

import json
import pytest
from unittest.mock import patch


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
    with patch("trafilatura.fetch_url", return_value="<html><body><article>Page content</article></body></html>"), \
         patch("trafilatura.extract", return_value="Page content"):
        assert search.extract("https://example.com") == "Page content"


def test_extract_download_failed():
    from src.backend import search
    with patch("trafilatura.fetch_url", return_value=None):
        assert search.extract("https://example.com") is None


def test_extract_import_error():
    from src.backend import search
    with patch.dict("sys.modules", {"trafilatura": None}):
        # trafilatura not installed
        pass
    with patch("builtins.__import__", side_effect=ImportError("no trafilatura")):
        assert search.extract("https://example.com") is None

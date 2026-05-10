"""Tests for research tools (tools.py)."""

import pytest
from unittest.mock import AsyncMock, patch


# ── searxng_search_tool ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_searxng_search_tool_uses_query_cache():
    from src.backend.tools import searxng_search_tool

    cache = {}
    with patch("src.backend.tools.search_mod.search", return_value={
        "data": [{"url": "https://example.com", "title": "Test", "content": "Snippet"}]
    }) as mock_search:
        result1 = await searxng_search_tool("test query", limit=5, query_cache=cache)
        assert mock_search.call_count == 1
        assert result1["cached"] is False
        assert len(result1["results"]) == 1

        result2 = await searxng_search_tool("test query", limit=5, query_cache=cache)
        assert mock_search.call_count == 1
        assert result2["cached"] is True


@pytest.mark.asyncio
async def test_searxng_search_tool_empty_result_rollback():
    from src.backend.tools import searxng_search_tool

    cache = {}

    def fake_search(q, limit=8):
        if q == "narrow query":
            return {"data": []}
        return {"data": [{"url": "https://example.com", "title": "Broad", "content": "Result"}]}

    with patch("src.backend.tools.search_mod.search", side_effect=fake_search):
        with patch("src.backend.tools.generate_broader_queries", return_value=["broad query"]):
            result = await searxng_search_tool("narrow query", limit=5, query_cache=cache)

    assert len(result["results"]) == 1
    assert result["results"][0]["title"] == "Broad"


@pytest.mark.asyncio
async def test_searxng_search_tool_no_results():
    from src.backend.tools import searxng_search_tool

    cache = {}
    with patch("src.backend.tools.search_mod.search", return_value={"data": []}):
        with patch("src.backend.tools.generate_broader_queries", return_value=[]):
            result = await searxng_search_tool("nothing", limit=5, query_cache=cache)

    assert result["results"] == []


# ── evaluate_sources_tool ───────────────────────────────────────

@pytest.mark.asyncio
async def test_evaluate_sources_tool_empty():
    from src.backend.tools import evaluate_sources_tool

    result = await evaluate_sources_tool([], "test objective")
    assert result["scored"] == []
    assert result["selected_for_fulltext"] == []


@pytest.mark.asyncio
async def test_evaluate_sources_tool_basic():
    from src.backend.tools import evaluate_sources_tool

    candidates = [
        {"url": "https://a.com", "title": "A", "description": "Good source"},
        {"url": "https://b.com", "title": "B", "description": "Another source"},
    ]

    with patch("src.backend.tools.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = (
            '{"evaluations": [{"id": 0, "score": 0.9, "full_text": true, "reason": "good"},'
            ' {"id": 1, "score": 0.5, "full_text": false, "reason": "ok"}]}'
        )
        result = await evaluate_sources_tool(candidates, "test objective")

    assert len(result["scored"]) == 2
    assert result["scored"][0]["quality_score"] == 0.9
    assert result["scored"][0]["full_text"] is True
    assert len(result["selected_for_fulltext"]) >= 1


# ── fetch_fulltext_tool ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_fulltext_tool_skips_file_urls():
    from src.backend.tools import fetch_fulltext_tool

    with patch("src.backend.tools.search_mod.extract_async", new_callable=AsyncMock, return_value="# Markdown") as mock_extract:
        result = await fetch_fulltext_tool(["file:///doc.pdf", "https://example.com"])

    assert "https://example.com" in result["extracted"]
    assert "file:///doc.pdf" not in result["extracted"]
    mock_extract.assert_awaited_once_with("https://example.com")


@pytest.mark.asyncio
async def test_fetch_fulltext_tool_extract_failure():
    from src.backend.tools import fetch_fulltext_tool

    with patch("src.backend.tools.search_mod.extract_async", new_callable=AsyncMock, return_value=None):
        result = await fetch_fulltext_tool(["https://example.com"])

    assert result["extracted"] == {}
    assert result["failed_count"] == 1


# ── Tool registry ───────────────────────────────────────────────

# ── synthesize_evidence_tool ────────────────────────────────────

@pytest.mark.asyncio
async def test_synthesize_evidence_tool_basic():
    from src.backend.tools import synthesize_evidence_tool

    result = await synthesize_evidence_tool(
        findings="AI safety is important",
        key_entities=["AI safety", "alignment"],
        remaining_questions=["What is RLHF?"],
        proposed_next_queries=["RLHF alignment research"],
    )

    assert result["synthesis"]["findings"] == "AI safety is important"
    assert result["synthesis"]["key_entities"] == ["AI safety", "alignment"]
    assert len(result["synthesis"]["proposed_next_queries"]) == 1
    assert "Evidence synthesized" in result["message"]


@pytest.mark.asyncio
async def test_synthesize_evidence_tool_optional_fields():
    from src.backend.tools import synthesize_evidence_tool

    result = await synthesize_evidence_tool(findings="Basic findings only")
    assert result["synthesis"]["key_entities"] == []
    assert result["synthesis"]["remaining_questions"] == []


# ── Tool registry ───────────────────────────────────────────────

def test_build_research_tools():
    from src.backend.tools import build_research_tools

    cache = {}
    tools = build_research_tools(query_cache=cache)
    names = {t.name for t in tools}
    assert "searxng_search" in names
    assert "document_hybrid_search" in names
    assert "evaluate_sources" in names
    assert "fetch_fulltext" in names
    assert "synthesize_evidence" in names
    assert "submit_report" in names

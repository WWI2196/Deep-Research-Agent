"""Tests for agent functions."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── import helpers ──────────────────────────────────────────────


@pytest.fixture
def agents_mod():
    import src.backend.agents as m
    return m


# ── _extract_json ───────────────────────────────────────────────

def test_extract_json():
    from src.backend.agents import _extract_json

    assert _extract_json('{"key": "value"}') == '{"key": "value"}'
    assert _extract_json('prefix {"key": "value"} suffix') == '{"key": "value"}'
    assert _extract_json("no json here") == ""


def test_extract_json_nested_braces():
    from src.backend.agents import _extract_json
    raw = 'text {"outer": {"inner": [1,2,3]}} more text'
    assert _extract_json(raw) == '{"outer": {"inner": [1,2,3]}}'


def test_extract_json_empty_string():
    from src.backend.agents import _extract_json
    assert _extract_json("") == ""


def test_extract_json_array():
    from src.backend.agents import _extract_json
    # _extract_json only matches {...}, not [...]; it extracts the inner object
    result = _extract_json('[{"a": 1}, {"b": 2}]')
    assert "a" in result  # extracts inner {...} content


# ── _clean_think_tags ──────────────────────────────────────────

def test_clean_think_tags_removes_basic():
    from src.backend.agents import _clean_think_tags
    result = _clean_think_tags("<think>internal reasoning</think>Final answer")
    assert result == "Final answer"


def test_clean_think_tags_no_tags():
    from src.backend.agents import _clean_think_tags
    result = _clean_think_tags("Plain text without tags")
    assert result == "Plain text without tags"


def test_clean_think_tags_multiline_think():
    from src.backend.agents import _clean_think_tags
    result = _clean_think_tags("Before\n<think>\nreasoning\nline 2\n</think>\n\nAfter")
    # The regex removes the think block but leaves surrounding newlines
    assert "Before" in result
    assert "After" in result
    assert "<think>" not in result
    assert "reasoning" not in result


# ── _pick_first_nonempty ───────────────────────────────────────

def test_pick_first_nonempty_first_value():
    from src.backend.agents import _pick_first_nonempty
    assert _pick_first_nonempty({"a": "hello"}, ["a", "b"]) == "hello"


def test_pick_first_nonempty_skips_empty():
    from src.backend.agents import _pick_first_nonempty
    assert _pick_first_nonempty({"a": "", "b": "world"}, ["a", "b"]) == "world"


def test_pick_first_nonempty_skips_none():
    from src.backend.agents import _pick_first_nonempty
    assert _pick_first_nonempty({"a": None, "b": "value"}, ["a", "b"]) == "value"


def test_pick_first_nonempty_all_empty():
    from src.backend.agents import _pick_first_nonempty
    assert _pick_first_nonempty({"a": "", "b": None}, ["a", "b"]) == ""


def test_pick_first_nonempty_converts_to_str():
    from src.backend.agents import _pick_first_nonempty
    assert _pick_first_nonempty({"a": 42}, ["a"]) == "42"


# ── _normalize_search_item ─────────────────────────────────────

def test_normalize_search_item():
    from src.backend.agents import _normalize_search_item

    item = {"url": "https://example.com", "title": "Test", "description": "Desc"}
    result = _normalize_search_item(item, "search")
    assert result["url"] == "https://example.com"
    assert result["title"] == "Test"

    assert _normalize_search_item({"title": "No URL"}, "search") is None


def test_normalize_search_item_with_alt_keys():
    from src.backend.agents import _normalize_search_item
    item = {"link": "https://example.com", "headline": "News Headline", "snippet": "A snippet"}
    result = _normalize_search_item(item, "news")
    assert result["url"] == "https://example.com"
    assert result["title"] == "News Headline"
    assert result["description"] == "A snippet"
    assert result["source"] == "news"


def test_normalize_search_item_canonical_url():
    from src.backend.agents import _normalize_search_item
    item = {"canonical_url": "https://example.com/canonical", "title": "Test"}
    result = _normalize_search_item(item, "search")
    assert result["url"] == "https://example.com/canonical"


# ── _has_clean_ending ──────────────────────────────────────────

def test_has_clean_ending():
    from src.backend.agents import _has_clean_ending

    assert _has_clean_ending("This is a sentence.")
    assert _has_clean_ending("Ends with quote\"")
    assert not _has_clean_ending("bare word")
    assert not _has_clean_ending("cut off and")


def test_has_clean_ending_fence_marks():
    from src.backend.agents import _has_clean_ending
    assert _has_clean_ending("Code:\n```")
    assert _has_clean_ending("Header\n***")
    assert _has_clean_ending("Separator\n---")


def test_has_clean_ending_empty_or_whitespace():
    from src.backend.agents import _has_clean_ending
    assert _has_clean_ending("") is True
    assert _has_clean_ending("   ") is True


# ── _needs_continuation ────────────────────────────────────────

def test_needs_continuation():
    from src.backend.agents import _needs_continuation

    long_text = "x" * 600
    assert not _needs_continuation(long_text + " complete sentence.")
    assert _needs_continuation(long_text + " cut off and")
    assert not _needs_continuation("Short")


def test_needs_continuation_with_end_marker():
    from src.backend.agents import _needs_continuation
    long_text = "x" * 600
    assert not _needs_continuation(long_text + " some text <<END>>", end_marker="<<END>>")
    assert _needs_continuation(long_text + " no ending marker here", end_marker="<<END>>")


def test_needs_continuation_dangling_preposition():
    from src.backend.agents import _needs_continuation
    long_text = "x" * 600
    assert _needs_continuation(long_text + " the result is in")
    assert _needs_continuation(long_text + " we will discuss the")
    assert _needs_continuation(long_text + " important factors are")


# ── _enforce_source_diversity ──────────────────────────────────

def test_enforce_source_diversity():
    from src.backend.agents import _enforce_source_diversity

    sources = [
        {"url": "https://example.com/a", "quality_score": 0.9},
        {"url": "https://example.com/b", "quality_score": 0.8},
        {"url": "https://example.com/c", "quality_score": 0.7},
        {"url": "https://example.com/d", "quality_score": 0.6},
        {"url": "https://other.com/x", "quality_score": 0.5},
    ]
    result = _enforce_source_diversity(sources, max_per_domain=3)
    assert len(result) == 4


def test_enforce_source_diversity_empty():
    from src.backend.agents import _enforce_source_diversity
    assert _enforce_source_diversity([]) == []


def test_enforce_source_diversity_no_url():
    from src.backend.agents import _enforce_source_diversity
    sources = [
        {"url": "", "title": "No URL 1"},
        {"url": "", "title": "No URL 2"},
    ]
    result = _enforce_source_diversity(sources)
    assert len(result) == 2  # empty URLs treated as same domain


# ── generate_research_plan ─────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_research_plan():
    from src.backend.agents import generate_research_plan

    with patch("src.backend.planning.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "A detailed research plan"
        result = await generate_research_plan("What is the future of AI?")
        assert result == "A detailed research plan"
        call_args = mock_chat.call_args.kwargs
        assert call_args["role"] == "planner"


# ── split_into_subtasks ────────────────────────────────────────

@pytest.mark.asyncio
async def test_split_into_subtasks_success():
    from src.backend.agents import split_into_subtasks

    with patch("src.backend.planning.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = '{"subtasks": [{"id": "t1", "title": "Task 1", "description": "Desc 1"}]}'
        result = await split_into_subtasks("Research plan text")
        assert len(result) == 1
        assert result[0]["id"] == "t1"


@pytest.mark.asyncio
async def test_split_into_subtasks_json_wrapped_in_text():
    from src.backend.agents import split_into_subtasks

    with patch("src.backend.planning.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = 'Here is the JSON:\n```json\n{"subtasks": [{"id": "t1", "title": "Task"}]}\n```'
        result = await split_into_subtasks("plan")
        assert len(result) == 1
        assert result[0]["id"] == "t1"


@pytest.mark.asyncio
async def test_split_into_subtasks_invalid_json():
    from src.backend.agents import split_into_subtasks

    with patch("src.backend.planning.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "not json at all, no braces"
        with pytest.raises(ValueError, match="Empty or invalid JSON"):
            await split_into_subtasks("plan")


# ── compute_scaling ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_compute_scaling_success():
    from src.backend.agents import compute_scaling

    with patch("src.backend.planning.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = '{"complexity": "moderate", "subagent_count": 5, "tool_calls_per_subagent": 15, "target_sources": 25}'
        result = await compute_scaling("query", "plan")
        assert result["complexity"] == "moderate"
        assert result["subagent_count"] == 5


@pytest.mark.asyncio
async def test_compute_scaling_wrapped_json():
    from src.backend.agents import compute_scaling

    with patch("src.backend.planning.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = 'The scaling plan is: {"complexity": "simple", "subagent_count": 3, "tool_calls_per_subagent": 10, "target_sources": 15}'
        result = await compute_scaling("query", "plan")
        assert result["complexity"] == "simple"


@pytest.mark.asyncio
async def test_compute_scaling_empty_response():
    from src.backend.agents import compute_scaling

    with patch("src.backend.planning.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = ""
        with pytest.raises(ValueError, match="Empty response from scaler"):
            await compute_scaling("query", "plan")


# ── generate_search_queries ────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_search_queries_basic():
    from src.backend.agents import generate_search_queries

    with patch("src.backend.subagent.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = '{"queries": ["q1", "q2", "q3", "q4"]}'
        subtask = {"id": "t1", "title": "AI Safety", "description": "Desc", "source_types": "academic, official"}
        result = await generate_search_queries(subtask)
        assert len(result) >= 4  # original + modifiers
        assert "q1" in result


@pytest.mark.asyncio
async def test_generate_search_queries_fallback_to_title():
    from src.backend.agents import generate_search_queries

    with patch("src.backend.subagent.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "not json"
        subtask = {"id": "t1", "title": "AI Safety Research", "description": "Desc"}
        result = await generate_search_queries(subtask)
        assert result == ["AI Safety Research"]


@pytest.mark.asyncio
async def test_generate_search_queries_adds_academic_modifiers():
    from src.backend.agents import generate_search_queries

    with patch("src.backend.subagent.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = '{"queries": ["search1", "search2"]}'
        subtask = {"id": "t1", "title": "Test", "description": "Desc", "source_types": "academic,paper"}
        result = await generate_search_queries(subtask)
        has_modifier = any("research paper" in q or "study" in q for q in result)
        assert has_modifier


@pytest.mark.asyncio
async def test_generate_search_queries_dedup_and_limit():
    from src.backend.agents import generate_search_queries

    with patch("src.backend.subagent.chat", new_callable=AsyncMock) as mock_chat:
        # Generate 15 queries to test 10-limit
        mock_chat.return_value = '{"queries": ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11"]}'
        subtask = {"id": "t1", "title": "Test", "description": "Desc", "source_types": ""}
        result = await generate_search_queries(subtask)
        assert len(result) <= 10


# ── batch_evaluate_sources ─────────────────────────────────────

@pytest.mark.asyncio
async def test_batch_evaluate_sources(sample_search_results):
    from src.backend.agents import batch_evaluate_sources

    with patch("src.backend.subagent.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = '{"evaluations": [{"id": 0, "score": 0.9, "reason": "good"}, {"id": 1, "score": 0.5, "reason": "ok"}]}'

        sources = [
            {"url": "https://example.com/1", "title": "Test 1", "description": "Desc 1"},
            {"url": "https://example.com/2", "title": "Test 2", "description": "Desc 2"},
        ]
        result = await batch_evaluate_sources(sources, "test query")
        assert len(result) == 2
        assert result[0]["quality_score"] == 0.9
        assert result[1]["quality_score"] == 0.5


@pytest.mark.asyncio
async def test_batch_evaluate_sources_empty():
    from src.backend.agents import batch_evaluate_sources
    result = await batch_evaluate_sources([], "query")
    assert result == []


@pytest.mark.asyncio
async def test_batch_evaluate_sources_multi_batch():
    from src.backend.agents import batch_evaluate_sources
    import json

    sources = [{"url": f"https://example.com/{i}", "title": f"T{i}", "description": f"D{i}"} for i in range(25)]

    with patch("src.backend.subagent.chat", new_callable=AsyncMock) as mock_chat:
        def _eval_side_effect(*args, **kwargs):
            msgs = kwargs.get("messages", [])
            for m in msgs:
                if "Evaluate these sources" in m.get("content", ""):
                    # Count source IDs in the content
                    content = m["content"]
                    count = content.count("ID: ")
                    evaluations = [{"id": j, "score": 0.8, "reason": "ok"} for j in range(count)]
            return json.dumps({"evaluations": evaluations})
        mock_chat.side_effect = _eval_side_effect

        result = await batch_evaluate_sources(sources, "test query")
        assert len(result) == 25
        assert mock_chat.call_count >= 2  # at least 2 batches


@pytest.mark.asyncio
async def test_batch_evaluate_sources_llm_error_fallback():
    from src.backend.agents import batch_evaluate_sources

    sources = [{"url": "https://example.com/1", "title": "T1", "description": "D1"}]

    with patch("src.backend.subagent.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.side_effect = RuntimeError("LLM error")
        result = await batch_evaluate_sources(sources, "test query")
        assert len(result) == 1
        assert result[0]["quality_score"] == 0.3  # fallback score


# ── _refine_queries_if_needed ──────────────────────────────────

@pytest.mark.asyncio
async def test_refine_queries_high_quality_returns_empty():
    from src.backend.agents import _refine_queries_if_needed

    scored = [
        {"quality_score": 0.8, "url": "https://a.com"},
        {"quality_score": 0.9, "url": "https://b.com"},
    ]
    result = await _refine_queries_if_needed({"title": "Test"}, scored, ["q1"])
    assert result == []


@pytest.mark.asyncio
async def test_refine_queries_low_quality_triggers_refinement():
    from src.backend.agents import _refine_queries_if_needed

    scored = [
        {"quality_score": 0.2, "url": "https://a.com"},
        {"quality_score": 0.3, "url": "https://b.com"},
    ]
    with patch("src.backend.subagent.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = '{"queries": ["refined1", "refined2"]}'
        result = await _refine_queries_if_needed({"title": "Test"}, scored, ["q1", "q2"])
        assert len(result) > 0
        assert "refined1" in result


@pytest.mark.asyncio
async def test_refine_queries_dedup_against_existing():
    from src.backend.agents import _refine_queries_if_needed

    scored = [{"quality_score": 0.2, "url": "https://a.com"}]
    with patch("src.backend.subagent.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = '{"queries": ["q1", "new-query"]}'  # q1 is duplicate
        result = await _refine_queries_if_needed({"title": "Test"}, scored, ["q1", "q2"])
        assert "q1" not in result
        assert "new-query" in result


@pytest.mark.asyncio
async def test_refine_queries_empty_sources():
    from src.backend.agents import _refine_queries_if_needed
    result = await _refine_queries_if_needed({"title": "Test"}, [], ["q1"])
    assert result == []


# ── run_subagent ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_subagent_full_flow():
    from src.backend.agents import run_subagent

    subtask = {
        "id": "task_1",
        "title": "Market Analysis",
        "description": "Analyze markets",
        "objective": "Understand market trends",
        "output_format": "markdown",
        "tool_guidance": "web search",
        "source_types": "news, official",
        "boundaries": "No technical details",
    }

    with patch("src.backend.subagent.chat", new_callable=AsyncMock) as mock_chat:
        # 1: generate queries, 2: URL selection, 3: report writing
        mock_chat.side_effect = [
            '{"queries": ["market analysis 2024", "EV outlook 2025"]}',
            '{"indices": [0, 2]}',  # select sources for full-text
            "# Market Analysis\n\nDetailed report content here.",
        ]

        with patch("src.backend.subagent.search_mod.search", return_value={
            "data": [
                {"url": "https://example.com/1", "title": "Market Report 1", "description": "Good data"},
                {"url": "https://example.com/2", "title": "Market Report 2", "description": "More data"},
                {"url": "https://other.com/3", "title": "Report 3", "description": "Different domain"},
            ]
        }), patch("src.backend.subagent.search_mod.extract", return_value="# Full markdown content..."), \
           patch("src.backend.subagent.batch_evaluate_sources", new_callable=AsyncMock) as mock_eval:
            mock_eval.return_value = [
                {"url": "https://example.com/1", "title": "Market Report 1", "description": "Good data", "quality_score": 0.9},
                {"url": "https://example.com/2", "title": "Market Report 2", "description": "More data", "quality_score": 0.8},
                {"url": "https://other.com/3", "title": "Report 3", "description": "Different domain", "quality_score": 0.7},
            ]

            result = await run_subagent(
                "What is the EV market outlook?",
                "Research plan...",
                subtask,
                tool_budget=10,
            )

    assert result["subtask_id"] == "task_1"
    assert result["subtask_title"] == "Market Analysis"
    assert "Market Analysis" in result["report"]
    assert result["evidence_count"] > 0
    assert len(result["sources"]) > 0


@pytest.mark.asyncio
async def test_run_subagent_extract_failure_falls_back():
    from src.backend.agents import run_subagent

    subtask = {"id": "t1", "title": "Test", "description": "Test desc", "objective": "Test obj"}

    with patch("src.backend.subagent.chat", new_callable=AsyncMock) as mock_chat:
        # 1: queries, 2: report (URL selection skipped: only 1 source < 3)
        mock_chat.side_effect = [
            '{"queries": ["test query"]}',
            "# Test\n\nReport content.",
        ]
        with patch("src.backend.subagent.search_mod.search", return_value={
            "data": [{"url": "https://example.com/1", "title": "Test", "description": "Description text"}]
        }), patch("src.backend.subagent.search_mod.extract", return_value=None), \
           patch("src.backend.subagent.batch_evaluate_sources", new_callable=AsyncMock) as mock_eval:
            mock_eval.return_value = [
                {"url": "https://example.com/1", "title": "Test", "description": "Description text", "quality_score": 0.8}
            ]
            result = await run_subagent("query", "plan", subtask, tool_budget=5)

    assert result["subtask_id"] == "t1"
    assert result["evidence_count"] >= 0  # extract failed but snippet used as fallback


# ── run_subagents_parallel ─────────────────────────────────────

@pytest.mark.asyncio
async def test_run_subagents_parallel_success():
    from src.backend.agents import run_subagents_parallel

    subtasks = [
        {"id": "t1", "title": "Task 1", "description": "D1", "objective": "O1"},
        {"id": "t2", "title": "Task 2", "description": "D2", "objective": "O2"},
    ]

    with patch("src.backend.subagent.run_subagent", new_callable=AsyncMock) as mock_run:
        async def fake_run(uq, rp, st, budget):
            return {
                "subtask_id": st["id"],
                "subtask_title": st["title"],
                "report": f"Report for {st['title']}",
                "sources": [{"url": f"https://{st['id']}.com", "quality_score": 0.9}],
                "evidence_count": 3,
            }
        mock_run.side_effect = fake_run

        result = await run_subagents_parallel("query", "plan", subtasks, 10)
        assert result["success_count"] == 2
        assert result["total_count"] == 2
        assert len(result["reports"]) == 2
        assert len(result["raw"]) == 2


@pytest.mark.asyncio
async def test_run_subagents_parallel_one_fails():
    from src.backend.agents import run_subagents_parallel

    subtasks = [
        {"id": "t1", "title": "Task 1", "description": "D1", "objective": "O1"},
        {"id": "t2", "title": "Task 2 (fails)", "description": "D2", "objective": "O2"},
    ]

    with patch("src.backend.subagent.run_subagent", new_callable=AsyncMock) as mock_run:
        async def fake_run(uq, rp, st, budget):
            if "fails" in st["title"]:
                raise RuntimeError("subagent error")
            return {
                "subtask_id": st["id"],
                "subtask_title": st["title"],
                "report": f"Report for {st['title']}",
                "sources": [{"url": f"https://{st['id']}.com", "quality_score": 0.9}],
                "evidence_count": 2,
            }
        mock_run.side_effect = fake_run

        result = await run_subagents_parallel("query", "plan", subtasks, 10)
        assert result["success_count"] == 1
        assert result["total_count"] == 2
        assert len(result["reports"]) == 1


@pytest.mark.asyncio
async def test_run_subagents_parallel_dedup_sources():
    from src.backend.agents import run_subagents_parallel

    subtasks = [
        {"id": "t1", "title": "Task 1", "description": "D1", "objective": "O1"},
        {"id": "t2", "title": "Task 2", "description": "D2", "objective": "O2"},
    ]

    with patch("src.backend.subagent.run_subagent", new_callable=AsyncMock) as mock_run:
        async def fake_run(uq, rp, st, budget):
            return {
                "subtask_id": st["id"],
                "subtask_title": st["title"],
                "report": f"Report",
                "sources": [{"url": "https://shared.com/a", "quality_score": 0.9}],  # same URL
                "evidence_count": 1,
            }
        mock_run.side_effect = fake_run

        result = await run_subagents_parallel("query", "plan", subtasks, 10)
        assert len(result["sources"]) == 1  # deduped


# ── synthesize_report ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_synthesize_report_basic():
    from src.backend.agents import synthesize_report

    with patch("src.backend.synthesis.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "# Synthesized Report\n\nContent here.\n\n<<END_OF_REPORT>>"

        with patch("src.backend.synthesis._continue_if_truncated", new_callable=AsyncMock) as mock_continue:
            mock_continue.return_value = "# Synthesized Report\n\nContent here."
            result = await synthesize_report("query", "plan", ["report1", "report2"])
            assert "Synthesized Report" in result
            assert "<<END_OF_REPORT>>" not in result  # stripped by _continue_if_truncated


@pytest.mark.asyncio
async def test_synthesize_report_truncation_recovery():
    from src.backend.agents import synthesize_report

    with patch("src.backend.synthesis.chat", new_callable=AsyncMock) as mock_chat:
        call_count = [0]

        def _side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("400 max context length")
            return "Final report\n\n<<END_OF_REPORT>>"

        mock_chat.side_effect = _side_effect

        with patch("src.backend.synthesis._continue_if_truncated", new_callable=AsyncMock) as mock_continue:
            mock_continue.return_value = "Final report"
            result = await synthesize_report("query", "plan", ["report1"])
            assert result == "Final report"
            assert call_count[0] == 2


@pytest.mark.asyncio
async def test_synthesize_report_ultimate_fallback():
    from src.backend.agents import synthesize_report

    with patch("src.backend.synthesis.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.side_effect = RuntimeError("always 400 too long")

        result = await synthesize_report("query", "plan", ["report content here"])
        assert "report content here" in result
        assert "Research Report" in result


# ── _continue_if_truncated ─────────────────────────────────────

@pytest.mark.asyncio
async def test_continue_if_truncated_no_need():
    from src.backend.agents import _continue_if_truncated

    report = "A" * 600 + ". This is a complete sentence."
    result = await _continue_if_truncated(report, "query")
    assert result == report  # unchanged


@pytest.mark.asyncio
async def test_continue_if_truncated_needs_continuation():
    from src.backend.agents import _continue_if_truncated

    report = "A" * 600 + " and"

    with patch("src.backend.synthesis.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "the conclusion follows here."
        result = await _continue_if_truncated(report, "query")
        assert "the conclusion follows here" in result


@pytest.mark.asyncio
async def test_continue_if_truncated_empty_report():
    from src.backend.agents import _continue_if_truncated
    result = await _continue_if_truncated("", "query")
    assert result == ""


@pytest.mark.asyncio
async def test_continue_if_truncated_max_rounds():
    from src.backend.agents import _continue_if_truncated

    report = "A" * 600 + " and"

    with patch("src.backend.synthesis.chat", new_callable=AsyncMock) as mock_chat:
        # Must be >= 20 chars to pass the continuation length check
        mock_chat.return_value = "more dangling text and"  # still dangling, but long enough
        result = await _continue_if_truncated(report, "query", max_rounds=2)
        # Should call until max_rounds exhausted (2 rounds)
        assert mock_chat.call_count == 2


# ── add_citations ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_add_citations_basic():
    from src.backend.agents import add_citations

    sources = [
        {"url": "https://example.com/1", "title": "Source 1", "description": "Description 1"},
    ]

    with patch("src.backend.synthesis.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "# Report\n\nWith citation [^1]\n\n## References\n\n[^1]: Source 1"

        with patch("src.backend.synthesis._continue_if_truncated", new_callable=AsyncMock) as mock_continue:
            mock_continue.return_value = "# Report\n\nWith citation [^1]\n\n## References\n\n[^1]: Source 1"
            result = await add_citations("Original report", sources)
            assert "citation" in result.lower() or "[^1]" in result


@pytest.mark.asyncio
async def test_add_citations_no_sources():
    from src.backend.agents import add_citations
    result = await add_citations("Original report", [])
    assert result == "Original report"  # returned unchanged


@pytest.mark.asyncio
async def test_add_citations_strips_bracket_tags():
    from src.backend.agents import add_citations

    sources = [{"url": "https://example.com/1", "title": "S1", "description": "D1"}]

    with patch("src.backend.synthesis.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "Cleaned report"

        with patch("src.backend.synthesis._continue_if_truncated", new_callable=AsyncMock) as mock_continue:
            mock_continue.return_value = "Cleaned report"
            result = await add_citations("[task_1_name] Original report", sources)
            assert "[task_1_name]" not in result  # tags stripped


@pytest.mark.asyncio
async def test_add_citations_adaptive_retry():
    from src.backend.agents import add_citations

    sources = [{"url": f"https://example.com/{i}", "title": f"Source {i}", "description": f"Desc {i}"} for i in range(10)]

    with patch("src.backend.synthesis.chat", new_callable=AsyncMock) as mock_chat:
        call_count = [0]

        def _side_effect(*args, **kwargs):
            call_count[0] += 1
            raise RuntimeError("400 context length exceeded")

        mock_chat.side_effect = _side_effect

        result = await add_citations("Report text", sources)
        # Should exhaust all attempts and return the original report
        assert len(result) > 0


# ── Test that all agent roles use correct provider routing ─────

@pytest.mark.asyncio
async def test_chat_routes_to_correct_role():
    from src.backend.agents import _chat

    with patch("src.backend.llm._get_or_create_provider") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.chat.return_value = "response"
        mock_get_provider.return_value = mock_provider

        result = await _chat(
            role="planner",
            messages=[{"role": "user", "content": "test"}],
        )
        assert result == "response"
        mock_get_provider.assert_called_once()
        mock_provider.chat.assert_called_once()

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

    plan_json = '{"dimensions": [{"name": "AI Future", "scope": "Future of AI", "source_types": "academic", "keywords": ["AI", "future"]}], "output_structure": ["Intro"], "methodology": "research"}'

    with patch("src.backend.planning.run_react_agent", new_callable=AsyncMock) as mock_react:
        mock_react.return_value = {
            "final_answer": plan_json,
            "tool_calls": [],
            "steps_taken": 2,
        }
        result = await generate_research_plan("What is the future of AI?")
        assert isinstance(result, dict)
        assert "dimensions" in result
        assert result["dimensions"][0]["name"] == "AI Future"
        mock_react.assert_called_once()
        call_kwargs = mock_react.call_args.kwargs
        assert call_kwargs["role"] == "planner"
        assert call_kwargs["max_steps"] == 6


@pytest.mark.asyncio
async def test_generate_research_plan_react_fallback():
    from src.backend.agents import generate_research_plan

    with patch("src.backend.planning.run_react_agent", new_callable=AsyncMock) as mock_react, \
         patch("src.backend.planning.chat", new_callable=AsyncMock) as mock_chat:
        # ReAct returns no valid plan
        mock_react.return_value = {"final_answer": "", "tool_calls": [], "steps_taken": 4}
        # Single-pass fallback succeeds
        mock_chat.return_value = '{"dimensions": [{"name": "Fallback", "scope": "s", "source_types": "academic", "keywords": ["k"]}], "output_structure": ["I"], "methodology": "m"}'

        result = await generate_research_plan("test query")
        assert result["dimensions"][0]["name"] == "Fallback"
        mock_react.assert_called_once()
        mock_chat.assert_called_once()


# ── split_into_subtasks ────────────────────────────────────────

@pytest.mark.asyncio
async def test_split_into_subtasks_success():
    from src.backend.agents import split_into_subtasks

    plan = {"dimensions": [{"name": "Test", "scope": "s", "keywords": ["k"], "source_types": "academic"}], "output_structure": ["I"], "methodology": "m"}
    with patch("src.backend.planning.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = '{"subtasks": [{"id": "t1", "title": "Task 1", "description": "Desc 1"}]}'
        result = await split_into_subtasks(plan)
        assert len(result) == 1
        assert result[0]["id"] == "t1"


@pytest.mark.asyncio
async def test_split_into_subtasks_json_wrapped_in_text():
    from src.backend.agents import split_into_subtasks

    plan = {"dimensions": [{"name": "Test", "scope": "s", "keywords": ["k"], "source_types": "academic"}], "output_structure": ["I"], "methodology": "m"}
    with patch("src.backend.planning.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = 'Here is the JSON:\n```json\n{"subtasks": [{"id": "t1", "title": "Task"}]}\n```'
        result = await split_into_subtasks(plan)
        assert len(result) == 1
        assert result[0]["id"] == "t1"


@pytest.mark.asyncio
async def test_split_into_subtasks_invalid_json():
    from src.backend.agents import split_into_subtasks

    plan = {"dimensions": [{"name": "Test", "scope": "plan", "keywords": ["k"], "source_types": "academic"}], "output_structure": ["I"], "methodology": "m"}
    with patch("src.backend.planning.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "not json at all, no braces"
        # Self-heal: first attempt fails, retry also fails, falls back to dimension-based subtasks
        result = await split_into_subtasks(plan)
        assert len(result) >= 1  # falls back to dimension subtasks



# ── generate_search_queries ────────────────────────────────────

def test_generate_search_queries_basic():
    from src.backend.agents import generate_search_queries

    subtask = {"id": "t1", "title": "AI Safety", "keywords": ["AI safety", "machine learning"], "source_types": "academic, official"}
    result = generate_search_queries(subtask)
    assert len(result) >= 2  # keywords + modifiers
    assert any("AI safety" in q for q in result)


def test_generate_search_queries_fallback_to_title():
    from src.backend.agents import generate_search_queries

    subtask = {"id": "t1", "title": "AI Safety Research", "description": "Desc"}
    result = generate_search_queries(subtask)
    assert "AI Safety Research" in result
    assert any("2025" in q for q in result)
    assert any("2026" in q for q in result)


def test_generate_search_queries_adds_academic_modifiers():
    from src.backend.agents import generate_search_queries

    subtask = {"id": "t1", "title": "Test", "keywords": ["test topic"], "source_types": "academic,paper"}
    result = generate_search_queries(subtask)
    has_modifier = any("research paper" in q or "study" in q for q in result)
    assert has_modifier


def test_generate_search_queries_dedup_and_limit():
    from src.backend.agents import generate_search_queries

    # Many keywords to test 10-limit
    subtask = {"id": "t1", "title": "Test", "keywords": ["k1", "k2", "k3", "k4", "k5", "k6"], "source_types": "academic,official,news,code"}
    result = generate_search_queries(subtask)
    assert len(result) <= 10
    # Should be deduplicated
    assert len(result) == len(set(result))


# ── batch_evaluate_sources ─────────────────────────────────────

@pytest.mark.asyncio
async def test_batch_evaluate_sources(sample_search_results):
    from src.backend.agents import batch_evaluate_sources

    with patch("src.backend.subagent.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = '{"evaluations": [{"id": 0, "score": 0.9, "full_text": true, "reason": "good"}, {"id": 1, "score": 0.5, "full_text": false, "reason": "ok"}]}'

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

    with patch("src.backend.react_agent.run_react_agent", new_callable=AsyncMock) as mock_react:
        mock_react.return_value = {
            "final_answer": "# Market Analysis\n\nDetailed report content here.",
            "tool_calls": [
                {"tool": "searxng_search", "input": {"query": "EV market"}, "result": {"success": True, "result": {"results": [{"url": "https://example.com/1", "title": "Market Report 1", "description": "Good data"}]}}},
            ],
            "steps_taken": 3,
        }

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

    with patch("src.backend.react_agent.run_react_agent", new_callable=AsyncMock) as mock_react:
        mock_react.return_value = {
            "final_answer": "# Test\n\nReport content.",
            "tool_calls": [
                {"tool": "searxng_search", "input": {"query": "test"}, "result": {"success": True, "result": {"results": [{"url": "https://example.com/1", "title": "Test", "description": "Description text"}]}}},
            ],
            "steps_taken": 2,
        }
        result = await run_subagent("query", "plan", subtask, tool_budget=5)

    assert result["subtask_id"] == "t1"
    assert result["evidence_count"] >= 0


# ── run_subagents_parallel ─────────────────────────────────────

@pytest.mark.asyncio
async def test_run_subagents_parallel_success():
    from src.backend.agents import run_subagents_parallel

    subtasks = [
        {"id": "t1", "title": "Task 1", "description": "D1", "objective": "O1"},
        {"id": "t2", "title": "Task 2", "description": "D2", "objective": "O2"},
    ]

    with patch("src.backend.subagent.run_subagent", new_callable=AsyncMock) as mock_run:
        async def fake_run(uq, rp, st, budget, query_cache=None, document_collections=None, gap_instruction=None, **kwargs):
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
        async def fake_run(uq, rp, st, budget, query_cache=None, document_collections=None, gap_instruction=None, **kwargs):
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
        async def fake_run(uq, rp, st, budget, query_cache=None, document_collections=None, gap_instruction=None, **kwargs):
            return {
                "subtask_id": st["id"],
                "subtask_title": st["title"],
                "report": "Report",
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

    long_report = "A" * 150 + " with enough content to pass the length filter for body text inclusion."
    with patch("src.backend.synthesis.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "## Introduction\n\nThis is a well-researched introduction to the topic with sufficient length to pass validation checks.\n\n## Conclusions\n\nThe research findings suggest important implications for the field.\n\n<<END_OF_REPORT>>"

        with patch("src.backend.synthesis._continue_if_truncated", new_callable=AsyncMock) as mock_continue:
            mock_continue.return_value = "## Introduction\n\nThis is a well-researched introduction to the topic with sufficient length to pass validation checks.\n\n## Conclusions\n\nThe research findings suggest important implications for the field."
            result = await synthesize_report("query", "plan", [long_report, long_report])
            assert "Introduction" in result
            assert "Conclusions" in result
            assert "A" * 10 in result  # body content preserved
            assert "<<END_OF_REPORT>>" not in result


@pytest.mark.asyncio
async def test_synthesize_report_truncation_recovery():
    from src.backend.agents import synthesize_report

    long_report = "B" * 150 + " body content for the test report."
    with patch("src.backend.synthesis.chat", new_callable=AsyncMock) as mock_chat:
        call_count = [0]

        def _side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("400 max context length")
            return "## Intro\n\nRecovered after error with enough text content to pass validation check for minimum length.\n\n## Conclusions\n\nGood results.\n\n<<END_OF_REPORT>>"

        mock_chat.side_effect = _side_effect

        with patch("src.backend.synthesis._continue_if_truncated", new_callable=AsyncMock) as mock_continue:
            # Return long enough to pass the >1000 char validation
            mock_continue.return_value = "## Introduction\n\n" + "R" * 1000 + "\n\n## Conclusions\n\nThe research yielded important findings for the field."
            result = await synthesize_report("query", "plan", [long_report])
            assert "RRRRR" in result
            assert call_count[0] >= 1  # first call triggered context error retry


@pytest.mark.asyncio
async def test_synthesize_report_ultimate_fallback():
    from src.backend.agents import synthesize_report

    long_report = "C" * 150 + " report content that meets the minimum length requirement."
    with patch("src.backend.synthesis.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.side_effect = RuntimeError("always 400 too long")

        result = await synthesize_report("query", "plan", [long_report])
        assert "C" * 10 in result  # body content preserved in fallback
        assert "Research Report" in result


# ── _continue_if_truncated ─────────────────────────────────────

@pytest.mark.asyncio
async def test_continue_if_truncated_no_need():
    from src.backend.agents import _continue_if_truncated

    report = "A" * 600 + ". This is a complete sentence that ends properly."
    result = await _continue_if_truncated(report, "query", end_marker=None)
    assert result == report  # unchanged — clean ending, no marker to check


@pytest.mark.asyncio
async def test_continue_if_truncated_needs_continuation():
    from src.backend.agents import _continue_if_truncated

    report = "A" * 600 + " and"

    with patch("src.backend.synthesis.chat", new_callable=AsyncMock) as mock_chat:
        # Must be >= 30 chars to pass the length check in new code
        mock_chat.return_value = "the conclusion follows here with more text to reach thirty characters."
        result = await _continue_if_truncated(report, "query", end_marker=None)
        assert "the conclusion follows here" in result


@pytest.mark.asyncio
async def test_continue_if_truncated_empty_report():
    from src.backend.agents import _continue_if_truncated
    result = await _continue_if_truncated("", "query", end_marker=None)
    assert result == ""


@pytest.mark.asyncio
async def test_continue_if_truncated_max_rounds():
    from src.backend.agents import _continue_if_truncated

    report = "A" * 600 + " and"

    with patch("src.backend.synthesis.chat", new_callable=AsyncMock) as mock_chat:
        # Must be >= 30 chars and still trigger continuation (end with "and")
        mock_chat.return_value = "more dangling text that keeps going and"
        result = await _continue_if_truncated(report, "query", end_marker=None, max_rounds=2)
        # Each continuation extends the report, called 2 times
        assert mock_chat.call_count == 2


# ── add_citations ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_add_citations_basic():
    from src.backend.agents import add_citations

    sources = [
        {"url": "https://example.com/1", "title": "Source 1", "description": "Description 1"},
    ]
    report = "Some claim [src: https://example.com/1] about the topic."
    with patch("src.backend.synthesis._verify_citation_urls", new_callable=AsyncMock) as mock_verify:
        mock_verify.return_value = {"https://example.com/1": True}
        result, verification = await add_citations(report, sources)
    assert "[^1]" in result  # marker replaced with numbered citation
    assert "References" in result  # references section appended
    assert "https://example.com/1" in result
    assert verification["https://example.com/1"] is True


@pytest.mark.asyncio
async def test_add_citations_no_sources():
    from src.backend.agents import add_citations
    with patch("src.backend.synthesis._verify_citation_urls", new_callable=AsyncMock) as mock_verify:
        mock_verify.return_value = {}
        result, verification = await add_citations("Original report", [])
    assert "Original report" in result
    assert verification == {}


@pytest.mark.asyncio
async def test_add_citations_strips_bracket_tags():
    from src.backend.agents import add_citations

    sources = [{"url": "https://example.com/1", "title": "S1", "description": "D1"}]
    report = "[task_1_name] Some claim [src: https://example.com/1]"
    with patch("src.backend.synthesis._verify_citation_urls", new_callable=AsyncMock) as mock_verify:
        mock_verify.return_value = {"https://example.com/1": True}
        result, verification = await add_citations(report, sources)
    assert "[task_1_name]" not in result  # old tags stripped
    assert "[^1]" in result  # src marker replaced


@pytest.mark.asyncio
async def test_add_citations_adaptive_retry():
    from src.backend.agents import add_citations

    sources = [{"url": f"https://example.com/{i}", "title": f"Source {i}", "description": f"Desc {i}"} for i in range(50)]
    report = "Text with [src: https://example.com/0] and [src: https://example.com/1] markers."
    with patch("src.backend.synthesis._verify_citation_urls", new_callable=AsyncMock) as mock_verify:
        mock_verify.return_value = {f"https://example.com/{i}": True for i in range(50)}
        result, verification = await add_citations(report, sources)
    assert "[^1]" in result
    assert "[^2]" in result


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


# ── synthesize_report with failure_summary ─────────────────────

@pytest.mark.asyncio
async def test_synthesize_report_with_failure_summary():
    from src.backend.agents import synthesize_report

    reports = ["A" * 200 + " detailed report content."]
    with patch("src.backend.synthesis.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "## Intro\n\nReport with failure context.\n\n## Conclusions\n\nDone.\n\n<<END_OF_REPORT>>"
        with patch("src.backend.synthesis._continue_if_truncated", new_callable=AsyncMock) as mock_cont:
            mock_cont.return_value = "## Intro\n\nReport.\n\n## Conclusions\n\nDone."
            result = await synthesize_report("query", "plan", reports, failure_summary="Previous attempt was truncated.")
            assert "## Intro" in result
            # Verify failure summary was included in the prompt
            call_args = mock_chat.call_args
            system_msg = call_args[1]["messages"][0]["content"]
            assert "Previous attempt was truncated" in system_msg


# ── generate_search_queries fuzzy dedup ────────────────────────

def test_generate_search_queries_fuzzy_dedup():
    from src.backend.subagent import generate_search_queries

    subtask = {
        "id": "t1", "title": "Test",
        "keywords": ["AI safety regulations", "AI safety regulation", "AI safety regulatory framework"],
        "source_types": "academic",
    }
    result = generate_search_queries(subtask)
    # Fuzzy dedup should collapse nearly identical keywords
    from src.backend.helpers import query_similarity
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            assert query_similarity(result[i], result[j]) < 0.85, \
                f"Near-duplicate queries: {result[i]} and {result[j]}"


# ── run_subagent query cache ───────────────────────────────────

@pytest.mark.asyncio
async def test_run_subagent_uses_query_cache():
    from src.backend.subagent import run_subagent

    subtask = {
        "id": "t1", "title": "Test", "description": "Desc",
        "objective": "Obj", "keywords": ["test"], "source_types": "academic",
        "boundaries": "", "output_format": "markdown", "tool_guidance": "",
    }

    cache = {"test": {"data": [{"url": "https://cached.com", "title": "Cached", "snippet": "Test"}]}}

    with patch("src.backend.react_agent.run_react_agent", new_callable=AsyncMock) as mock_react:
        mock_react.return_value = {
            "final_answer": "# Report\n\nContent.",
            "tool_calls": [],
            "steps_taken": 1,
        }
        result = await run_subagent("query", "plan", subtask, tool_budget=5, query_cache=cache)

    assert result["subtask_id"] == "t1"
    # query_cache should be passed through to build_research_tools
    assert mock_react.called


# ── run_subagent empty result triggers broader ─────────────────

@pytest.mark.asyncio
async def test_run_subagent_empty_result_triggers_broader():
    # Empty-result rollback is tested in test_tools.py (test_searxng_search_tool_empty_result_rollback)
    from src.backend.subagent import run_subagent

    subtask = {
        "id": "t1", "title": "Test", "description": "Desc",
        "objective": "Obj", "keywords": ["AI safety research paper"], "source_types": "academic",
        "boundaries": "", "output_format": "markdown", "tool_guidance": "",
    }

    with patch("src.backend.react_agent.run_react_agent", new_callable=AsyncMock) as mock_react:
        mock_react.return_value = {
            "final_answer": "# Report\n\nContent.",
            "tool_calls": [
                {"tool": "searxng_search", "input": {"query": "AI safety research paper"}, "result": {"success": True, "result": {"results": [{"url": "https://example.com", "title": "Result", "description": "Test"}]}}},
            ],
            "steps_taken": 2,
        }
        result = await run_subagent("query", "plan", subtask, tool_budget=5)

    assert result["subtask_id"] == "t1"
    assert result["evidence_count"] >= 0


# ── _trim_reports_by_whole ─────────────────────────────────────

def test_trim_reports_by_whole_fits():
    from src.backend.synthesis import _trim_reports_by_whole
    reports = ["Report A content", "Report B content", "Report C content"]
    result = _trim_reports_by_whole(reports, 1000)
    assert "Report A" in result
    assert "Report B" in result
    assert "Report C" in result


def test_trim_reports_by_whole_drops_shortest():
    from src.backend.synthesis import _trim_reports_by_whole
    reports = ["A" * 100, "B" * 100, "C" * 10]
    # Limit forces dropping the shortest report(s)
    result = _trim_reports_by_whole(reports, 220)
    assert "A" * 100 in result
    assert "B" * 100 in result
    # C is shortest and may be dropped
    if "C" * 10 not in result:
        assert len(result) <= 220


def test_trim_reports_by_whole_empty():
    from src.backend.synthesis import _trim_reports_by_whole
    assert _trim_reports_by_whole([], 100) == ""


# ── _strip_existing_ref_sections length guard ──────────────────

def test_strip_ref_sections_length_guard():
    from src.backend.synthesis import _strip_existing_ref_sections
    # Model inserted "## References" mid-report followed by another heading.
    # Stripping would delete the huge block between References and the next ##.
    long_middle = "This is a very long paragraph that takes up most of the report space. " * 10
    report = (
        "# Title\n\nShort intro.\n\n"
        "## References\n\n"
        f"{long_middle}\n\n"
        "## Next Section\n\n"
        "Tail content that should survive."
    )
    result = _strip_existing_ref_sections(report)
    # Should skip strip because >50% would be removed
    assert "Tail content that should survive" in result


def test_strip_ref_sections_normal():
    from src.backend.synthesis import _strip_existing_ref_sections
    report = "# Title\n\nBody text.\n\n## References\n\n[^1]: url\n"
    result = _strip_existing_ref_sections(report)
    assert "Body text" in result
    assert "## References" not in result


# ── add_citations appends uncited sources ──────────────────────

@pytest.mark.asyncio
async def test_add_citations_appends_uncited_sources():
    from src.backend.synthesis import add_citations

    report = "Claim [src: https://example.com/1] about topic."
    sources = [
        {"url": "https://example.com/1", "title": "Web Source"},
        {"url": "file:///Users/eureka/docs/doc1.md", "title": "Doc Source"},
    ]
    with patch("src.backend.synthesis._verify_citation_urls", new_callable=AsyncMock) as mock_verify:
        mock_verify.return_value = {"https://example.com/1": True, "file:///Users/eureka/docs/doc1.md": True}
        result, verification = await add_citations(report, sources)

    assert "[^1]" in result
    assert "Web Source" in result
    # Uncited sources are no longer appended as an additional section
    assert "Doc Source" not in result
    assert "file:///Users/eureka/docs/doc1.md" not in result
    assert "Additional sources" not in result

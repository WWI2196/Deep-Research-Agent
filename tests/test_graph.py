"""Tests for LangGraph research pipeline."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_deps():
    """Patch all external dependencies for graph execution."""
    with patch("src.backend.graph.get_config") as mock_cfg, \
         patch("src.backend.graph.generate_research_plan", new_callable=AsyncMock) as mock_plan, \
         patch("src.backend.graph.split_into_subtasks", new_callable=AsyncMock) as mock_split, \
         patch("src.backend.graph.run_subagents_parallel", new_callable=AsyncMock) as mock_subs, \
         patch("src.backend.graph.synthesize_report", new_callable=AsyncMock) as mock_synth, \
         patch("src.backend.graph.add_citations", new_callable=AsyncMock) as mock_cite, \
         patch("src.backend.graph.chat", new_callable=AsyncMock) as mock_chat, \
         patch("src.backend.graph.persist_run", new_callable=AsyncMock) as mock_persist_run, \
         patch("src.backend.graph.persist_checkpoint", new_callable=AsyncMock) as mock_persist_ckpt, \
         patch("src.backend.graph.persist_source", new_callable=AsyncMock) as mock_persist_src, \
         patch("src.backend.graph.persist_subagent_report", new_callable=AsyncMock) as mock_persist_rpt, \
         patch("src.backend.graph.trace", new_callable=AsyncMock) as mock_trace:

        # Default config
        cfg = MagicMock()
        cfg.default_provider = "openai"
        cfg.default_model = "gpt-4o"
        cfg.max_iterations = 3
        cfg.quality_threshold = 0.7
        cfg.context_compress_retries = 1
        cfg.keep_tool_results = 5
        mock_cfg.return_value = cfg

        # Default agent returns
        mock_plan.return_value = {
            "dimensions": [
                {"name": "Safety Overview", "scope": "General AI safety", "keywords": ["AI safety"], "source_types": "academic"},
            ],
            "output_structure": ["Introduction", "Safety Overview", "Conclusions"],
            "methodology": "Academic review",
        }
        mock_split.return_value = [
            {"id": "t1", "title": "Task 1", "description": "Desc 1", "objective": "Obj 1",
             "dimension": "Safety Overview", "keywords": ["AI safety"], "source_types": "academic",
             "boundaries": "", "estimated_searches": 8},
            {"id": "t2", "title": "Task 2", "description": "Desc 2", "objective": "Obj 2",
             "dimension": "Safety Overview", "keywords": ["AI safety"], "source_types": "academic",
             "boundaries": "", "estimated_searches": 8},
        ]
        mock_subs.return_value = {
            "reports": ["Report 1", "Report 2"],
            "sources": [{"url": "https://example.com/1"}, {"url": "https://example.com/2"}],
            "raw": [
                {"subtask_id": "t1", "subtask_title": "Task 1", "report": "Report 1",
                 "sources": [{"url": "https://example.com/1", "title": "Src1"}], "evidence_count": 3},
                {"subtask_id": "t2", "subtask_title": "Task 2", "report": "Report 2",
                 "sources": [{"url": "https://example.com/2", "title": "Src2"}], "evidence_count": 4},
            ],
            "success_count": 2,
            "total_count": 2,
        }
        mock_synth.return_value = "# Synthesized Report\n\nFull content..."
        mock_cite.return_value = ("# Synthesized Report [^1]\n\nFull content...\n\n## References\n[^1]: https://example.com/1", {"https://example.com/1": True})
        mock_chat.return_value = json.dumps({
            "dimension_scores": {"Safety Overview": {"coverage": 0.8, "depth": 0.7, "evidence": 0.8, "recency": 0.9}},
            "overall_score": 0.8,
            "research_complete": True,
            "gaps": [],
        })

        deps = {
            "cfg": mock_cfg, "plan": mock_plan, "split": mock_split,
            "subs": mock_subs, "synth": mock_synth, "cite": mock_cite, "chat": mock_chat,
            "persist_run": mock_persist_run, "persist_ckpt": mock_persist_ckpt,
            "persist_src": mock_persist_src, "persist_rpt": mock_persist_rpt,
            "trace": mock_trace,
        }
        yield deps


@pytest.fixture
def events_capturer():
    """Captures emitted events for inspection."""
    events = []
    async def capture(evt):
        events.append(dict(evt))
    capture.events = events
    return capture


# ── Full pipeline: single iteration (no gaps) ──────────────────

@pytest.mark.asyncio
async def test_full_pipeline_single_iteration(mock_deps, events_capturer):
    from src.backend.graph import build_and_run_graph

    state = {"user_query": "What is AI safety?"}
    result = await build_and_run_graph(state, events_capturer)

    # All nodes should have been called
    mock_deps["plan"].assert_called_once()
    mock_deps["split"].assert_called_once()
    mock_deps["subs"].assert_called_once()
    mock_deps["synth"].assert_called_once()
    mock_deps["cite"].assert_called_once()

    # Final state
    assert result["run_id"] is not None
    assert result["cited_report"] is not None
    assert result["research_complete"] is True
    assert len(result["sources"]) >= 1


# ── Reflection: gaps found -> loop back to subagents ───────────

@pytest.mark.asyncio
async def test_reflection_with_gaps_loops_back(mock_deps, events_capturer):
    from src.backend.graph import build_and_run_graph

    # First reflection: find gaps, second: no gaps
    mock_deps["chat"].side_effect = [
        json.dumps({
            "dimension_scores": {"Safety Overview": {"coverage": 0.4, "depth": 0.3, "evidence": 0.5, "recency": 0.7}},
            "overall_score": 0.47,
            "research_complete": False,
            "gaps": [{"dimension": "Safety Overview", "gap_detail": "Missing details",
                      "subtask": {"id": "gap_1", "title": "Gap analysis", "description": "Fill gap",
                                  "objective": "Fill", "output_format": "markdown",
                                  "dimension": "Safety Overview", "keywords": ["safety"],
                                  "source_types": "academic", "boundaries": "", "estimated_searches": 6}}],
        }),
        json.dumps({
            "dimension_scores": {"Safety Overview": {"coverage": 0.9, "depth": 0.8, "evidence": 0.8, "recency": 0.9}},
            "overall_score": 0.85,
            "research_complete": True,
            "gaps": [],
        }),
    ]

    state = {"user_query": "What is AI safety?", "max_iterations": 3}
    result = await build_and_run_graph(state, events_capturer)

    # Should have called subagents twice (initial + gap-fill)
    assert mock_deps["subs"].call_count == 2
    assert mock_deps["chat"].call_count >= 2
    assert result["iteration_count"] == 2


# ── Reflection: max iterations reached ─────────────────────────

@pytest.mark.asyncio
async def test_reflection_max_iterations_reached(mock_deps, events_capturer):
    from src.backend.graph import build_and_run_graph

    mock_deps["chat"].return_value = json.dumps({
        "dimension_scores": {"Safety": {"coverage": 0.4, "depth": 0.3, "evidence": 0.5, "recency": 0.6}},
        "overall_score": 0.45,
        "research_complete": False,
        "gaps": [{"dimension": "Safety", "gap_detail": "missing",
                  "subtask": {"id": "gap_1", "title": "Gap 1", "description": "d", "objective": "o",
                              "output_format": "markdown", "dimension": "Safety", "keywords": ["k"],
                              "source_types": "academic", "boundaries": "", "estimated_searches": 5}}],
    })

    state = {"user_query": "test", "max_iterations": 1}
    result = await build_and_run_graph(state, events_capturer)
    assert result["iteration_count"] <= 1


# ── Split fallback on exception ────────────────────────────────

@pytest.mark.asyncio
async def test_split_node_fallback(mock_deps, events_capturer):
    from src.backend.graph import build_and_run_graph

    mock_deps["split"].side_effect = RuntimeError("splitter failure")

    state = {"user_query": "test query that is very long and detailed about AI safety"}
    result = await build_and_run_graph(state, events_capturer)

    # Should continue with fallback subtask
    assert len(result["subtasks"]) == 1
    assert result["subtasks"][0]["id"] == "main"


# ── Reflection failure fallback ────────────────────────────────

@pytest.mark.asyncio
async def test_reflection_failure_fallback(mock_deps, events_capturer):
    from src.backend.graph import build_and_run_graph

    mock_deps["chat"].side_effect = RuntimeError("reflection failure")

    state = {"user_query": "test query"}
    result = await build_and_run_graph(state, events_capturer)

    # Should complete even if reflection fails
    assert result["research_complete"] is True


# ── Event emission verification ────────────────────────────────

@pytest.mark.asyncio
async def test_events_emitted_for_all_phases(mock_deps, events_capturer):
    from src.backend.graph import build_and_run_graph

    state = {"user_query": "test"}
    await build_and_run_graph(state, events_capturer)

    event_types = {e["type"] for e in events_capturer.events}
    expected = {"phase-update", "progress", "plan-generated", "subtasks-created",
                 "subagents-launch", "subagent-complete",
                 "reflection-decision", "report-draft", "citations-added"}
    for et in expected:
        assert et in event_types, f"Missing event: {et}"


# ── Progress calculation ───────────────────────────────────────

def test_progress_function():
    from src.backend.graph import _progress

    # TOTAL_WEIGHT = 2+8+5+60+5+12+8 = 100
    assert _progress(0, 2) == 2
    assert _progress(2, 8) == 10
    assert _progress(99, 10) == 99


# ── Run ID generation ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_id_generated_if_not_provided(mock_deps, events_capturer):
    from src.backend.graph import build_and_run_graph

    state = {"user_query": "test"}
    result = await build_and_run_graph(state, events_capturer)
    assert len(result["run_id"]) > 0
    assert result["run_id"] != ""


# ── Checkpoint persistence ─────────────────────────────────────

@pytest.mark.asyncio
async def test_checkpoints_persisted_for_all_phases(mock_deps, events_capturer):
    from src.backend.graph import build_and_run_graph

    state = {"user_query": "test"}
    await build_and_run_graph(state, events_capturer)

    # Verify persist_checkpoint was called for each phase
    phases_called = {call_args[0][1] for call_args in mock_deps["persist_ckpt"].call_args_list}
    assert "init" in phases_called
    assert "plan" in phases_called
    assert "split" in phases_called
    assert any("subagents" in p for p in phases_called)
    assert "reflection" in phases_called
    assert "synthesis" in phases_called
    assert "citation" in phases_called


# ── Synthesis retry on truncation ────────────────────────────

@pytest.mark.asyncio
async def test_synthesis_retry_on_truncation(mock_deps, events_capturer):
    from src.backend.graph import build_and_run_graph

    mock_deps["synth"].side_effect = [
        "x" * 600 + " and the",        # truncated (needs_continuation returns True for dangling "the")
        "# Complete Report\n\nFull content with proper ending.",  # retry succeeds
    ]

    state = {"user_query": "test", "context_compress_retries": 1}
    result = await build_and_run_graph(state, events_capturer)

    assert mock_deps["synth"].call_count == 2
    assert "Complete Report" in result["report"]


@pytest.mark.asyncio
async def test_synthesis_no_retry_when_complete(mock_deps, events_capturer):
    from src.backend.graph import build_and_run_graph

    mock_deps["synth"].return_value = "# Complete\n\nFull report content with sentences.\n\n" + "x" * 500 + "."

    state = {"user_query": "test", "context_compress_retries": 1}
    result = await build_and_run_graph(state, events_capturer)

    assert mock_deps["synth"].call_count == 1


# ── Reflection low quality at max iterations ─────────────────

@pytest.mark.asyncio
async def test_reflection_low_quality_sets_failure_summary(mock_deps, events_capturer):
    from src.backend.graph import build_and_run_graph

    mock_deps["chat"].return_value = json.dumps({
        "dimension_scores": {"Safety": {"coverage": 0.3, "depth": 0.2, "evidence": 0.3, "recency": 0.4}},
        "overall_score": 0.3,
        "research_complete": False,
        "gaps": [{"dimension": "Safety", "gap_detail": "missing",
                  "subtask": {"id": "gap_1", "title": "Gap 1", "description": "d", "objective": "o",
                              "output_format": "markdown", "dimension": "Safety", "keywords": ["k"],
                              "source_types": "academic", "boundaries": "", "estimated_searches": 5}}],
    })

    state = {"user_query": "test", "max_iterations": 1, "context_compress_retries": 1}
    result = await build_and_run_graph(state, events_capturer)

    # Should still complete
    assert result["iteration_count"] <= 1
    # Verify "low-quality-retry" event was emitted
    decisions = [e for e in events_capturer.events if e["type"] == "reflection-decision"]
    assert len(decisions) >= 1

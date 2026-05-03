"""Tests for LangGraph research pipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

@pytest.fixture
def mock_deps():
    """Patch all external dependencies for graph execution."""
    with patch("src.backend.graph.get_config") as mock_cfg, \
         patch("src.backend.graph.generate_research_plan", new_callable=AsyncMock) as mock_plan, \
         patch("src.backend.graph.split_into_subtasks", new_callable=AsyncMock) as mock_split, \
         patch("src.backend.graph.compute_scaling", new_callable=AsyncMock) as mock_scale, \
         patch("src.backend.graph.run_subagents_parallel", new_callable=AsyncMock) as mock_subs, \
         patch("src.backend.graph.synthesize_report", new_callable=AsyncMock) as mock_synth, \
         patch("src.backend.graph.add_citations", new_callable=AsyncMock) as mock_cite, \
         patch("src.backend.graph.chat", new_callable=AsyncMock) as mock_chat, \
         patch("src.backend.graph.persist_run", new_callable=AsyncMock) as mock_persist_run, \
         patch("src.backend.graph.persist_checkpoint", new_callable=AsyncMock) as mock_persist_ckpt, \
         patch("src.backend.graph.persist_source", new_callable=AsyncMock) as mock_persist_src, \
         patch("src.backend.graph.persist_subagent_report", new_callable=AsyncMock) as mock_persist_rpt, \
         patch("src.backend.graph.update_run_status", new_callable=AsyncMock) as mock_update:

        # Default config
        cfg = MagicMock()
        cfg.default_provider = "openai"
        cfg.default_model = "gpt-4o"
        cfg.max_iterations = 3
        cfg.quality_threshold = 0.7
        mock_cfg.return_value = cfg

        # Default agent returns
        mock_plan.return_value = "Research plan: investigate AI safety..."
        mock_split.return_value = [
            {"id": "t1", "title": "Task 1", "description": "Desc 1", "objective": "Obj 1"},
            {"id": "t2", "title": "Task 2", "description": "Desc 2", "objective": "Obj 2"},
        ]
        mock_scale.return_value = {"complexity": "moderate", "subagent_count": 2, "tool_calls_per_subagent": 15, "target_sources": 10}
        mock_subs.return_value = {
            "reports": ["Report 1", "Report 2"],
            "sources": [{"url": "https://example.com/1"}, {"url": "https://example.com/2"}],
            "raw": [
                {"subtask_id": "t1", "subtask_title": "Task 1", "report": "Report 1", "sources": [{"url": "https://example.com/1"}], "evidence_count": 3},
                {"subtask_id": "t2", "subtask_title": "Task 2", "report": "Report 2", "sources": [{"url": "https://example.com/2"}], "evidence_count": 4},
            ],
            "success_count": 2,
            "total_count": 2,
        }
        mock_synth.return_value = "# Synthesized Report\n\nFull content..."
        mock_cite.return_value = "# Synthesized Report [^1]\n\nFull content...\n\n## References\n[^1]: https://example.com/1"
        mock_chat.return_value = '{"subtasks": []}'  # reflection: no gaps

        deps = {
            "cfg": mock_cfg, "plan": mock_plan, "split": mock_split, "scale": mock_scale,
            "subs": mock_subs, "synth": mock_synth, "cite": mock_cite, "chat": mock_chat,
            "persist_run": mock_persist_run, "persist_ckpt": mock_persist_ckpt, "update": mock_update,
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
    mock_deps["scale"].assert_called_once()
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
    import json

    # First reflection: find gaps
    mock_deps["chat"].side_effect = [
        json.dumps({"subtasks": [{"id": "gap_1", "title": "Gap analysis", "description": "Fill gap"}]}),
        json.dumps({"subtasks": []})  # second reflection: no gaps
    ]

    state = {"user_query": "What is AI safety?", "max_iterations": 3}
    result = await build_and_run_graph(state, events_capturer)

    # Should have called subagents twice (initial + gap-fill)
    assert mock_deps["subs"].call_count == 2
    assert mock_deps["chat"].call_count >= 2  # reflection called at least twice
    assert result["iteration_count"] == 2


# ── Reflection: max iterations reached ─────────────────────────

@pytest.mark.asyncio
async def test_reflection_max_iterations_reached(mock_deps, events_capturer):
    from src.backend.graph import build_and_run_graph
    import json

    # Reflection always finds gaps, but max_iterations limits it
    mock_deps["chat"].return_value = json.dumps({
        "subtasks": [{"id": "gap_1", "title": "Gap 1"}]
    })

    state = {"user_query": "test", "max_iterations": 1}
    result = await build_and_run_graph(state, events_capturer)

    # Should stop after first iteration because max_iterations=1
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


# ── Scale fallback on exception ────────────────────────────────

@pytest.mark.asyncio
async def test_scale_node_fallback(mock_deps, events_capturer):
    from src.backend.graph import build_and_run_graph

    mock_deps["scale"].side_effect = RuntimeError("scaler failure")

    state = {"user_query": "test query"}
    result = await build_and_run_graph(state, events_capturer)

    # Should continue with fallback scaling
    scaling = result.get("scaling", {})
    assert isinstance(scaling, dict)


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
                 "scaling-computed", "subagents-launch", "subagent-complete",
                 "reflection-decision", "report-draft", "citations-added"}
    for et in expected:
        assert et in event_types, f"Missing event: {et}"


# ── Progress calculation ───────────────────────────────────────

def test_progress_function():
    from src.backend.graph import _progress

    w = sum([2, 8, 5, 5, 55, 5, 12, 8])  # TOTAL_WEIGHT
    # After init (2)
    assert _progress(0, 2) == 2  # (0+2)/100 * 100 = 2
    # After init + plan (2 + 8)
    assert _progress(2, 8) == 10
    # Capped at 99
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
    result = await build_and_run_graph(state, events_capturer)

    # Verify persist_checkpoint was called for each phase
    phases_called = {call_args[0][1] for call_args in mock_deps["persist_ckpt"].call_args_list}
    # Should have init, plan, split, scale, subagents, reflection, synthesis, citation
    assert "init" in phases_called
    assert "plan" in phases_called
    assert "split" in phases_called
    assert "scale" in phases_called
    assert any("subagents" in p for p in phases_called)
    assert "reflection" in phases_called
    assert "synthesis" in phases_called
    assert "citation" in phases_called

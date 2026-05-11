"""Tests for structured tracing and LLM call logging."""

import pytest
import uuid
from unittest.mock import patch

from src.backend.persistence import init_db, get_run_logs, get_run_llm_calls, get_run_timeline
from src.backend.tracing import current_run_id, trace, trace_llm_call


@pytest.fixture(autouse=True)
def ensure_db():
    init_db()


@pytest.fixture
def run_id():
    return f"run_{uuid.uuid4().hex[:8]}"


def _mock_cfg(log_level="info"):
    """Return a minimal mock config with the given log_level."""
    cfg = type("C", (), {
        "log_level": log_level,
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
    })()
    return cfg


@pytest.mark.asyncio
async def test_trace_with_run_id(run_id):
    token = current_run_id.set(run_id)
    try:
        with patch("src.backend.tracing.get_config", return_value=_mock_cfg("info")):
            row_id = await trace("plan", "node_enter", "Entering plan", {"dim": 3})
        assert row_id is not None
        logs = get_run_logs(run_id)
        assert len(logs) == 1
        assert logs[0]["phase"] == "plan"
        assert logs[0]["event_type"] == "node_enter"
        assert logs[0]["details"] is not None
    finally:
        current_run_id.reset(token)


@pytest.mark.asyncio
async def test_trace_without_run_id():
    current_run_id.set(None)
    with patch("src.backend.tracing.get_config", return_value=_mock_cfg("info")):
        row_id = await trace("plan", "node_enter", "Entering plan")
    assert row_id is None


@pytest.mark.asyncio
async def test_trace_llm_call_with_run_id(run_id):
    token = current_run_id.set(run_id)
    try:
        with patch("src.backend.tracing.get_config", return_value=_mock_cfg("info")):
            await trace_llm_call(
                role="planner",
                messages=[{"role": "user", "content": "hello"}],
                provider="openai",
                model="gpt-4o",
                response="plan result",
                latency_ms=1200,
            )
        calls = get_run_llm_calls(run_id)
        assert len(calls) == 1
        assert calls[0]["role"] == "planner"
        assert calls[0]["provider"] == "openai"
        assert calls[0]["model"] == "gpt-4o"
        assert calls[0]["latency_ms"] == 1200
        # In info mode response is truncated to 200 chars
        assert "plan result" in calls[0]["response"]
    finally:
        current_run_id.reset(token)


@pytest.mark.asyncio
async def test_trace_llm_call_without_run_id():
    current_run_id.set(None)
    with patch("src.backend.tracing.get_config", return_value=_mock_cfg("info")):
        await trace_llm_call(
            role="planner",
            messages=[{"role": "user", "content": "hello"}],
            provider="openai",
            model="gpt-4o",
        )


@pytest.mark.asyncio
async def test_debug_trace_filtered_when_info_level(run_id):
    """Debug-level traces should be skipped when config is info."""
    token = current_run_id.set(run_id)
    try:
        with patch("src.backend.tracing.get_config", return_value=_mock_cfg("info")):
            await trace("plan", "node_enter", "info enter")
            await trace("plan", "debug_detail", "debug stuff", level="debug")
            await trace("split", "node_enter", "split enter")

        logs = get_run_logs(run_id)
        assert len(logs) == 2
        assert all(l["message"] != "debug stuff" for l in logs)
    finally:
        current_run_id.reset(token)


@pytest.mark.asyncio
async def test_debug_trace_persisted_when_debug_level(run_id):
    """Debug-level traces should be persisted when config is debug."""
    token = current_run_id.set(run_id)
    try:
        with patch("src.backend.tracing.get_config", return_value=_mock_cfg("debug")):
            await trace("plan", "node_enter", "info enter")
            await trace("plan", "debug_detail", "debug stuff", level="debug")

        logs = get_run_logs(run_id)
        assert len(logs) == 2
        messages = [l["message"] for l in logs]
        assert "info enter" in messages
        assert "debug stuff" in messages
    finally:
        current_run_id.reset(token)


@pytest.mark.asyncio
async def test_get_run_logs_filtered(run_id):
    token = current_run_id.set(run_id)
    try:
        with patch("src.backend.tracing.get_config", return_value=_mock_cfg("info")):
            await trace("plan", "node_enter", "plan enter")
            await trace("plan", "node_exit", "plan exit")
            await trace("split", "node_enter", "split enter")
            await trace("split", "error", "split err", level="warning")

        all_logs = get_run_logs(run_id)
        assert len(all_logs) == 4

        plan_logs = get_run_logs(run_id, phase="plan")
        assert len(plan_logs) == 2

        warning_logs = get_run_logs(run_id, level="warning")
        assert len(warning_logs) == 1
        assert warning_logs[0]["event_type"] == "error"

        enter_logs = get_run_logs(run_id, event_type="node_enter")
        assert len(enter_logs) == 2
    finally:
        current_run_id.reset(token)


@pytest.mark.asyncio
async def test_get_run_llm_calls_filtered(run_id):
    token = current_run_id.set(run_id)
    try:
        with patch("src.backend.tracing.get_config", return_value=_mock_cfg("info")):
            await trace_llm_call(role="planner", messages=[], provider="o", model="m")
            await trace_llm_call(role="subagent", messages=[], provider="o", model="m")
            await trace_llm_call(role="subagent", messages=[], provider="o", model="m")

        all_calls = get_run_llm_calls(run_id)
        assert len(all_calls) == 3

        planner_calls = get_run_llm_calls(run_id, role="planner")
        assert len(planner_calls) == 1

        sub_calls = get_run_llm_calls(run_id, role="subagent")
        assert len(sub_calls) == 2
    finally:
        current_run_id.reset(token)


@pytest.mark.asyncio
async def test_get_run_timeline_merged(run_id):
    token = current_run_id.set(run_id)
    try:
        with patch("src.backend.tracing.get_config", return_value=_mock_cfg("info")):
            await trace("plan", "node_enter", "enter")
            await trace_llm_call(role="planner", messages=[], provider="o", model="m", latency_ms=100)
            await trace("plan", "node_exit", "exit")

        items = get_run_timeline(run_id)
        assert len(items) == 3
        types = [i["type"] for i in items]
        assert "node_enter" in types
        assert "llm_call" in types
        assert "node_exit" in types
        llm_item = [i for i in items if i["type"] == "llm_call"][0]
        assert llm_item["latency_ms"] == 100
    finally:
        current_run_id.reset(token)

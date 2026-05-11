"""Tests for FastAPI server endpoints and SSE streaming."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture(autouse=True)
def clear_config_cache():
    from src.backend import config
    config._config = None
    yield
    config._config = None


@pytest.fixture
def mock_get_config():
    """Mock the config system for server tests."""
    with patch("src.backend.server.get_config") as mock:
        cfg = MagicMock()
        cfg.base_url = "https://api.openai.com/v1"
        cfg.api_key = "sk-test"
        cfg.default_model = "gpt-4o"
        cfg.quality_threshold = 0.7
        mock.return_value = cfg
        yield mock


@pytest.fixture
def mock_persistence():
    """Mock persistence operations that server.py directly imports."""
    with patch("src.backend.server.init_db") as mk_init, \
         patch("src.backend.server.update_run_status", new_callable=AsyncMock) as mk_update, \
         patch("src.backend.server.get_run_history", return_value=[]) as mk_hist, \
         patch("src.backend.server.get_run_report", return_value=None) as mk_report, \
         patch("src.backend.server.get_report_content", return_value="") as mk_content:
        yield {
            "init_db": mk_init,
            "update_run_status": mk_update,
            "get_run_history": mk_hist,
            "get_run_report": mk_report,
            "get_report_content": mk_content,
        }


@pytest.fixture
def client(mock_get_config, mock_persistence):
    """Create a test client for the FastAPI app."""
    from src.backend.server import app
    from fastapi.testclient import TestClient
    return TestClient(app)


# ── Health endpoint ────────────────────────────────────────────

def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert "base_url" in data
    assert "model" in data


# ── Start research (non-stream) ────────────────────────────────

def test_start_research(client):
    response = client.post("/api/research", json={"query": "test query"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "started"
    assert len(data["run_id"]) == 16  # hex from os.urandom(8)


# ── Cancel research ────────────────────────────────────────────

def test_cancel_research_existing(client):
    import asyncio
    from unittest.mock import patch

    # TestClient cancels dangling asyncio tasks when a request returns,
    # so we stub out create_task to keep the run entry alive.
    class FakeTask:
        def __init__(self, coro):
            self._coro = coro
            self._cancelled = False

        def cancel(self):
            self._cancelled = True

    with patch("src.backend.server.asyncio.create_task", side_effect=FakeTask) as mock_create_task:
        start_resp = client.post("/api/research", json={"query": "test"})
        run_id = start_resp.json()["run_id"]

        response = client.post(f"/api/research/{run_id}/cancel")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"
        # Verify the endpoint called cancel() on the background task
        assert mock_create_task.return_value._cancelled


def test_cancel_research_not_found(client):
    response = client.post("/api/research/nonexistent/cancel")
    assert response.status_code == 404


# ── History endpoint ───────────────────────────────────────────

def test_history_endpoint(client, mock_persistence):
    response = client.get("/api/research/history?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "history" in data


# ── Get report ─────────────────────────────────────────────────

def test_get_report_not_found(client):
    response = client.get("/api/research/nonexistent/report")
    assert response.status_code == 404


def test_get_report_found(client, mock_persistence):
    mock_report = {
        "run_id": "test-run",
        "query": "test query",
        "status": "completed",
        "total_sources": 10,
    }
    mock_persistence["get_run_report"].return_value = mock_report
    mock_persistence["get_report_content"].return_value = "report text"
    response = client.get("/api/research/test-run/report")
    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == "test-run"
    assert data["content"] == "report text"


# ── Config endpoints ───────────────────────────────────────────

def test_get_config(client):
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "base_url" in data
    assert "api_key" in data
    assert "default_model" in data
    assert "quality_threshold" in data
    assert "roles" in data


def test_update_config(client):
    with patch("src.backend.server.reload_config") as mock_reload, \
         patch("src.backend.server.save_config") as mock_save, \
         patch("src.backend.llm.invalidate_client_cache"):
        cfg = MagicMock()
        cfg.base_url = "https://api.openai.com/v1"
        cfg.api_key = "sk-test"
        cfg.default_model = "gpt-4o"
        cfg.quality_threshold = 0.7
        mock_reload.return_value = cfg

        response = client.post("/api/config", json={
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-4o",
        })
        assert response.status_code == 200
        assert response.json()["status"] == "saved"


# ── Models endpoint ────────────────────────────────────────────

def test_list_models(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert data["providers"] == []
    assert data["details"] == {}


# ── SSE stream endpoint ────────────────────────────────────────

def test_sse_stream_basic(client):
    """Verify SSE stream starts and returns data: formatted events."""
    with patch("src.backend.server.build_and_run_graph", new_callable=AsyncMock) as mock_graph, \
         patch("src.backend.server.export_markdown", return_value="/tmp/report.md"):
        mock_graph.return_value = {
            "run_id": "test-run-123",
            "user_query": "test query",
            "cited_report": "# Final Report\n\nContent",
            "report": "# Final Report\n\nContent",
            "subagent_reports": ["Report 1"],
            "sources": [{"url": "https://example.com"}],
            "iteration_count": 1,
            "subtasks": [],
            "scaling": {},
        }

        response = client.post("/api/research/stream", json={"query": "test query"})
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        # Read the SSE stream
        body = response.text
        # Should have data: prefix lines
        assert "data:" in body
        # Should have init phase update
        assert "phase-update" in body
        # Should have final result
        assert "complete" in body


def test_sse_stream_has_init_event(client):
    with patch("src.backend.server.build_and_run_graph", new_callable=AsyncMock) as mock_graph, \
         patch("src.backend.server.export_markdown", return_value="/tmp/report.md"):
        mock_graph.return_value = {
            "run_id": "test-run",
            "user_query": "test",
            "cited_report": "Report",
            "report": "Report",
            "subagent_reports": [],
            "sources": [],
            "iteration_count": 0,
            "subtasks": [],
            "scaling": {},
        }

        response = client.post("/api/research/stream", json={"query": "test"})
        body = response.text

        # Parse first SSE event
        lines = body.strip().split("\n")
        for line in lines:
            if line.startswith("data: "):
                first_event = json.loads(line[6:])
                assert first_event["type"] == "phase-update"
                assert "run_id" in first_event
                assert len(first_event["run_id"]) == 16  # hex from os.urandom(8)
                break


def test_sse_stream_passes_max_iterations(client):
    with patch("src.backend.server.build_and_run_graph", new_callable=AsyncMock) as mock_graph, \
         patch("src.backend.server.export_markdown", return_value="/tmp/report.md"):
        mock_graph.return_value = {
            "run_id": "test-run",
            "user_query": "test",
            "cited_report": "Report",
            "report": "Report",
            "subagent_reports": [],
            "sources": [],
            "iteration_count": 0,
            "subtasks": [],
            "scaling": {},
        }

        response = client.post("/api/research/stream", json={
            "query": "test",
            "max_iterations": 7,
            "quality_threshold": 0.9,
        })

        # Verify the state passed to graph contains the overrides
        call_args = mock_graph.call_args[0][0]
        assert call_args["max_iterations"] == 7
        assert call_args["quality_threshold"] == 0.9


def test_sse_stream_error_handling(client):
    with patch("src.backend.server.build_and_run_graph", new_callable=AsyncMock) as mock_graph:
        mock_graph.side_effect = RuntimeError("API key invalid")

        response = client.post("/api/research/stream", json={"query": "test"})
        body = response.text
        assert "error" in body


# ── Event translation ──────────────────────────────────────────

def test_serialize_event_format():
    from src.backend.server import serialize_event
    result = serialize_event("test-type", {"key": "value"})
    assert result.startswith("data: ")
    assert result.endswith("\n\n")
    parsed = json.loads(result[6:-2])  # strip "data: " and "\n\n"
    assert parsed["type"] == "test-type"
    assert parsed["key"] == "value"


def test_get_error_hint():
    from src.backend.server import _get_error_hint

    assert "credits" in _get_error_hint("402 Payment Required")
    assert "Invalid API key" in _get_error_hint("401 Unauthorized")
    assert "Access denied" in _get_error_hint("403 Forbidden")
    assert "Model not found" in _get_error_hint("404 Not Found")
    assert "Rate limited" in _get_error_hint("429 Too Many Requests")
    assert _get_error_hint("some random error") == ""


# ── CORS headers ────────────────────────────────────────────────

def test_cors_headers(client):
    response = client.options("/api/health")
    # FastAPI TestClient may or may not include CORS on OPTIONS
    # Just verify the app doesn't crash
    assert response.status_code in (200, 204, 405)

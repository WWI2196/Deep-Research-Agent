"""Integration tests — cross-module interactions."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture(autouse=True)
def clear_config_cache():
    from src.backend import config
    config._config = None
    yield
    config._config = None


# ── Config round-trip ──────────────────────────────────────────

def test_config_save_and_reload_api():
    """Integration: save config via API, reload, verify values."""
    from src.backend.config import get_config, load_config, AppConfig, RoleConfig

    with patch.dict("os.environ", {}, clear=True):
        cfg = AppConfig(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            default_model="gpt-4o",
            quality_threshold=0.85,
        )
        cfg.roles["planner"] = RoleConfig(model="gpt-4o")

        from src.backend.server import app
        from fastapi.testclient import TestClient

        with patch("src.backend.server.init_db"), \
             patch("src.backend.server.get_config", return_value=cfg), \
             patch("src.backend.server.reload_config", return_value=cfg), \
             patch("src.backend.server.save_config") as mock_save, \
             patch("src.backend.llm.invalidate_client_cache"):
            client = TestClient(app)

            resp = client.post("/api/config", json={
                "base_url": "https://api.openai.com/v1",
                "default_model": "gpt-4o-mini",
            })
            assert resp.status_code == 200
            assert resp.json()["status"] == "saved"
            mock_save.assert_called_once()


# ── Persistence + History flow ─────────────────────────────────

@pytest.mark.asyncio
async def test_persistence_complete_run_flow():
    """Integration: persist a run, record checkpoints, update status, retrieve history."""
    import src.backend.persistence as pmod
    import tempfile
    from pathlib import Path

    original_path = pmod.DB_PATH
    original_dir = pmod.DB_DIR

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            pmod.DB_DIR = Path(tmpdir)
            pmod.DB_PATH = Path(tmpdir) / "history.db"
            pmod.init_db()

            # Simulate a full research run
            run_id = "integration-run-1"
            await pmod.persist_run(run_id, "What is quantum computing?", "https://api.openai.com/v1", "gpt-4o")

            # Checkpoints at each phase
            phases = ["init", "plan", "split", "scale", "subagents", "reflection", "synthesis", "citation"]
            for i, phase in enumerate(phases):
                await pmod.persist_checkpoint(run_id, phase, {
                    "phase": phase, "step": i, "subtasks": [{"id": f"t{i}", "title": f"Task {i}"}]
                })

            # Persist some sources
            for j in range(5):
                await pmod.persist_source(run_id, f"https://example.com/{j}", title=f"Source {j}", quality_score=0.8, domain="example.com", subtask_id="t1")

            # Persist subagent reports
            await pmod.persist_subagent_report(run_id, "t1", "# Report 1\n\nContent", sources_count=3, evidence_count=5)
            await pmod.persist_subagent_report(run_id, "t2", "# Report 2\n\nContent", sources_count=2, evidence_count=4)

            # Complete the run
            await pmod.update_run_status(run_id, "completed", total_sources=5, total_reports=2, iterations=1)

            # Retrieve history
            history = pmod.get_run_history(limit=10)
            assert len(history) == 1
            assert history[0]["run_id"] == run_id
            assert history[0]["status"] == "completed"
            assert history[0]["total_sources"] == 5
            assert history[0]["total_reports"] == 2

            # Retrieve full report
            report = pmod.get_run_report(run_id)
            assert report is not None
            assert report["query"] == "What is quantum computing?"
    finally:
        pmod.DB_DIR = original_dir
        pmod.DB_PATH = original_path


# ── Search → Agent integration ─────────────────────────────────

@pytest.mark.asyncio
async def test_search_to_agent_integration():
    """Integration: SearXNG search returns data that agent can consume."""
    from src.backend import search as search_mod
    from src.backend.agents import _normalize_search_item, batch_evaluate_sources

    mock_response = json.dumps({
        "query": "AI research",
        "results": [
            {"title": "Research on AI", "url": "https://example.com/1", "content": "Comprehensive AI research paper", "score": 5.0, "engines": ["google"]},
            {"title": "ML Study", "url": "https://academic.example.com/2", "content": "Machine learning study results", "score": 4.0, "engines": ["bing"]},
            {"title": "News Article", "url": "https://example.com/3", "content": "Latest AI news", "score": 1.0, "engines": ["duckduckgo"]},
        ]
    })

    with patch("src.backend.search.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value.read.return_value = mock_response.encode()
        result = search_mod.search("AI research", limit=5)

    assert len(result["data"]) == 3

    # Normalize search results
    normalized = []
    for item in result["data"]:
        n = _normalize_search_item(item, "search")
        if n:
            normalized.append(n)

    assert len(normalized) == 3

    # Batch evaluate
    with patch("src.backend.subagent.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = json.dumps({
            "evaluations": [
                {"id": 0, "score": 0.9, "reason": "Academic paper, highly relevant"},
                {"id": 1, "score": 0.7, "reason": "Relevant but secondary"},
                {"id": 2, "score": 0.3, "reason": "News article, low depth"},
            ]
        })
        scored = await batch_evaluate_sources(normalized, "AI research")
        assert len(scored) == 3
        assert scored[0]["quality_score"] == 0.9
        assert scored[2]["quality_score"] == 0.3

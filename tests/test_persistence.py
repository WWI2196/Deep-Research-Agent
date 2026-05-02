"""Tests for SQLite persistence layer."""

import tempfile
import time
from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def temp_db():
    """Use a temp database file for each test."""
    import src.backend.persistence as pmod
    original_path = pmod.DB_PATH
    original_dir = pmod.DB_DIR

    with tempfile.TemporaryDirectory() as tmpdir:
        pmod.DB_DIR = Path(tmpdir)
        pmod.DB_PATH = Path(tmpdir) / "history.db"
        pmod.init_db()
        yield
        pmod.DB_DIR = original_dir
        pmod.DB_PATH = original_path


def test_init_db_creates_tables():
    import src.backend.persistence as pmod
    pmod.init_db()
    conn = pmod._get_conn()
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    conn.close()
    table_names = [t["name"] for t in tables]
    assert "runs" in table_names
    assert "checkpoints" in table_names
    assert "sources" in table_names
    assert "subagent_reports" in table_names


@pytest.mark.asyncio
async def test_persist_run_and_get_history():
    import src.backend.persistence as pmod
    await pmod.persist_run("run-1", "test query", "openai", "gpt-4o")

    history = pmod.get_run_history(limit=10)
    assert len(history) == 1
    assert history[0]["run_id"] == "run-1"
    assert history[0]["query"] == "test query"
    assert history[0]["status"] == "running"
    assert history[0]["provider"] == "openai"
    assert history[0]["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_update_run_status():
    import src.backend.persistence as pmod
    await pmod.persist_run("run-2", "query", "openai", "gpt-4o")
    await pmod.update_run_status(
        "run-2", "completed",
        total_sources=10,
        total_reports=3,
        iterations=2,
        report_path="/tmp/report.md",
    )

    report = pmod.get_run_report("run-2")
    assert report is not None
    assert report["status"] == "completed"
    assert report["total_sources"] == 10
    assert report["total_reports"] == 3
    assert report["iterations"] == 2
    assert report["report_path"] == "/tmp/report.md"
    assert report["completed_at"] is not None


@pytest.mark.asyncio
async def test_persist_checkpoint():
    import src.backend.persistence as pmod
    await pmod.persist_run("run-3", "query", "openai", "gpt-4o")

    state = {
        "run_id": "run-3",
        "user_query": "test",
        "subtasks": [{"id": "t1", "title": "Task 1"}],
        "sources": [{"url": "https://example.com"}],
    }
    await pmod.persist_checkpoint("run-3", "plan", state)

    conn = pmod._get_conn()
    rows = conn.execute(
        "SELECT * FROM checkpoints WHERE run_id=? AND phase=?",
        ("run-3", "plan"),
    ).fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0]["phase"] == "plan"


@pytest.mark.asyncio
async def test_persist_checkpoint_serializes_unserializable():
    import src.backend.persistence as pmod
    await pmod.persist_run("run-4", "query", "openai", "gpt-4o")

    class Unserializable:
        pass

    state = {
        "run_id": "run-4",
        "unserializable_obj": Unserializable(),
        "good_key": "good_value",
    }
    await pmod.persist_checkpoint("run-4", "split", state)

    conn = pmod._get_conn()
    rows = conn.execute(
        "SELECT state FROM checkpoints WHERE run_id=? AND phase=?",
        ("run-4", "split"),
    ).fetchall()
    conn.close()
    assert len(rows) >= 1


@pytest.mark.asyncio
async def test_persist_source():
    import src.backend.persistence as pmod
    await pmod.persist_run("run-5", "query", "openai", "gpt-4o")
    await pmod.persist_source(
        "run-5",
        url="https://example.com",
        title="Example",
        quality_score=0.9,
        domain="example.com",
        subtask_id="t1",
    )

    conn = pmod._get_conn()
    rows = conn.execute(
        "SELECT * FROM sources WHERE run_id=?",
        ("run-5",),
    ).fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0]["url"] == "https://example.com"
    assert rows[0]["title"] == "Example"
    assert rows[0]["quality_score"] == 0.9


@pytest.mark.asyncio
async def test_persist_subagent_report():
    import src.backend.persistence as pmod
    await pmod.persist_run("run-6", "query", "openai", "gpt-4o")
    await pmod.persist_subagent_report(
        "run-6",
        subtask_id="t1",
        content="# Report\n\nContent",
        sources_count=5,
        evidence_count=3,
    )

    conn = pmod._get_conn()
    rows = conn.execute(
        "SELECT * FROM subagent_reports WHERE run_id=?",
        ("run-6",),
    ).fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0]["subtask_id"] == "t1"
    assert rows[0]["sources_count"] == 5
    assert rows[0]["evidence_count"] == 3


@pytest.mark.asyncio
async def test_get_run_history_limit():
    import src.backend.persistence as pmod
    for i in range(5):
        await pmod.persist_run(f"run-{i}", f"query {i}", "openai", "gpt-4o")

    history = pmod.get_run_history(limit=3)
    assert len(history) == 3


def test_get_run_report_not_found():
    import src.backend.persistence as pmod
    report = pmod.get_run_report("nonexistent-run")
    assert report is None

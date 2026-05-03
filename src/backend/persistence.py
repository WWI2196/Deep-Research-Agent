"""SQLite persistence — research runs, checkpoints, sources, subagent reports."""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

DB_DIR = Path.home() / ".deep-research"
DB_PATH = DB_DIR / "history.db"


def _get_conn() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'running',
            provider TEXT,
            model TEXT,
            config_snapshot TEXT,
            total_sources INTEGER DEFAULT 0,
            total_reports INTEGER DEFAULT 0,
            iterations INTEGER DEFAULT 0,
            started_at INTEGER NOT NULL,
            completed_at INTEGER,
            report_path TEXT
        );
        CREATE TABLE IF NOT EXISTS checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            phase TEXT NOT NULL,
            state TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );
        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            url TEXT NOT NULL,
            title TEXT,
            quality_score REAL,
            domain TEXT,
            subtask_id TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );
        CREATE TABLE IF NOT EXISTS subagent_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            subtask_id TEXT NOT NULL,
            content TEXT NOT NULL,
            sources_count INTEGER,
            evidence_count INTEGER,
            created_at INTEGER NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );
        CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at DESC);
        CREATE INDEX IF NOT EXISTS idx_checkpoints_run_id ON checkpoints(run_id);
        CREATE INDEX IF NOT EXISTS idx_sources_run_id ON sources(run_id);
        CREATE INDEX IF NOT EXISTS idx_reports_run_id ON subagent_reports(run_id);
    """)
    conn.commit()
    conn.close()


async def persist_run(run_id: str, query: str, provider: str, model: str) -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO runs (run_id, query, status, provider, model, started_at) VALUES (?, ?, 'running', ?, ?, ?)",
        (run_id, query, provider, model, int(time.time())),
    )
    conn.commit()
    conn.close()


async def update_run_status(
    run_id: str,
    status: str,
    total_sources: int = 0,
    total_reports: int = 0,
    iterations: int = 0,
    report_path: str = "",
) -> None:
    conn = _get_conn()
    conn.execute(
        "UPDATE runs SET status=?, total_sources=?, total_reports=?, iterations=?, completed_at=?, report_path=? WHERE run_id=?",
        (status, total_sources, total_reports, iterations, int(time.time()), report_path, run_id),
    )
    conn.commit()
    conn.close()


async def persist_checkpoint(run_id: str, phase: str, state: dict[str, Any]) -> None:
    conn = _get_conn()
    serializable = {}
    for k, v in state.items():
        try:
            json.dumps(v)
            serializable[k] = v
        except (TypeError, ValueError):
            serializable[k] = str(v)

    conn.execute(
        "INSERT INTO checkpoints (run_id, phase, state, created_at) VALUES (?, ?, ?, ?)",
        (run_id, phase, json.dumps(serializable), int(time.time())),
    )
    conn.commit()
    conn.close()


async def persist_source(run_id: str, url: str, title: str = "", quality_score: float = 0.0, domain: str = "", subtask_id: str = "") -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT INTO sources (run_id, url, title, quality_score, domain, subtask_id) VALUES (?, ?, ?, ?, ?, ?)",
        (run_id, url, title, quality_score, domain, subtask_id),
    )
    conn.commit()
    conn.close()


async def persist_subagent_report(run_id: str, subtask_id: str, content: str, sources_count: int = 0, evidence_count: int = 0) -> None:
    conn = _get_conn()
    conn.execute(
        "INSERT INTO subagent_reports (run_id, subtask_id, content, sources_count, evidence_count, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (run_id, subtask_id, content, sources_count, evidence_count, int(time.time())),
    )
    conn.commit()
    conn.close()


def get_run_history(limit: int = 20) -> list[dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT run_id, query, status, provider, model, total_sources, total_reports, iterations, started_at, completed_at, report_path FROM runs ORDER BY started_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_run_report(run_id: str) -> dict[str, Any] | None:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_report_content(run_id: str) -> str:
    """Get the full report text for a completed run. Tries file first, then DB."""
    run = get_run_report(run_id)
    if not run:
        return ""

    # Try reading from the exported markdown file
    report_path = run.get("report_path", "")
    if report_path:
        try:
            return Path(report_path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            pass

    # Fall back to subagent reports in database
    conn = _get_conn()
    rows = conn.execute(
        "SELECT content FROM subagent_reports WHERE run_id=? ORDER BY created_at",
        (run_id,),
    ).fetchall()
    conn.close()
    if rows:
        return "\n\n".join(r["content"] for r in rows)

    return ""

"""SQLite persistence — research runs, checkpoints, sources, subagent reports.

All database I/O runs via asyncio.to_thread to avoid blocking the event loop.
"""

import asyncio
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


def _write_sync(op):
    """Run a write operation on a fresh connection with commit+close."""
    conn = _get_conn()
    try:
        result = op(conn)
        conn.commit()
        return result
    finally:
        conn.close()


def _read_sync(op):
    """Run a read operation on a fresh connection, returning the result."""
    conn = _get_conn()
    try:
        return op(conn)
    finally:
        conn.close()


async def _write_async(op):
    return await asyncio.to_thread(_write_sync, op)


async def _read_async(op):
    return await asyncio.to_thread(_read_sync, op)


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
        CREATE TABLE IF NOT EXISTS collections (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            doc_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'ready',
            created_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            collection_id TEXT NOT NULL,
            name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT,
            category TEXT,
            tags TEXT,
            page_count INTEGER DEFAULT 0,
            chunk_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'indexed',
            error_msg TEXT,
            created_at INTEGER NOT NULL,
            FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at DESC);
        CREATE INDEX IF NOT EXISTS idx_checkpoints_run_id ON checkpoints(run_id);
        CREATE INDEX IF NOT EXISTS idx_sources_run_id ON sources(run_id);
        CREATE INDEX IF NOT EXISTS idx_reports_run_id ON subagent_reports(run_id);
        CREATE INDEX IF NOT EXISTS idx_documents_collection_id ON documents(collection_id);
        CREATE TABLE IF NOT EXISTS llm_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            call_id TEXT NOT NULL,
            role TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            temperature REAL,
            max_tokens INTEGER,
            messages TEXT NOT NULL,
            response TEXT,
            latency_ms INTEGER,
            retry_attempt INTEGER DEFAULT 0,
            error TEXT,
            created_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trace_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            phase TEXT NOT NULL,
            level TEXT NOT NULL DEFAULT 'info',
            event_type TEXT NOT NULL,
            message TEXT NOT NULL,
            details TEXT,
            parent_id INTEGER,
            created_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_llm_calls_run_id ON llm_calls(run_id);
        CREATE INDEX IF NOT EXISTS idx_llm_calls_role ON llm_calls(role);
        CREATE INDEX IF NOT EXISTS idx_trace_logs_run_id ON trace_logs(run_id);
        CREATE INDEX IF NOT EXISTS idx_trace_logs_phase ON trace_logs(phase);
        CREATE INDEX IF NOT EXISTS idx_trace_logs_event_type ON trace_logs(event_type);
    """)
    conn.commit()
    conn.close()


async def persist_run(run_id: str, query: str, provider: str, model: str) -> None:
    await _write_async(lambda conn: conn.execute(
        "INSERT OR REPLACE INTO runs (run_id, query, status, provider, model, started_at) VALUES (?, ?, 'running', ?, ?, ?)",
        (run_id, query, provider, model, int(time.time())),
    ))


async def update_run_status(
    run_id: str,
    status: str,
    total_sources: int = 0,
    total_reports: int = 0,
    iterations: int = 0,
    report_path: str = "",
) -> None:
    await _write_async(lambda conn: conn.execute(
        "UPDATE runs SET status=?, total_sources=?, total_reports=?, iterations=?, completed_at=?, report_path=? WHERE run_id=?",
        (status, total_sources, total_reports, iterations, int(time.time()), report_path, run_id),
    ))


async def persist_checkpoint(run_id: str, phase: str, state: dict[str, Any]) -> None:
    serializable = {}
    for k, v in state.items():
        try:
            json.dumps(v)
            serializable[k] = v
        except (TypeError, ValueError):
            serializable[k] = str(v)

    await _write_async(lambda conn: conn.execute(
        "INSERT INTO checkpoints (run_id, phase, state, created_at) VALUES (?, ?, ?, ?)",
        (run_id, phase, json.dumps(serializable), int(time.time())),
    ))


async def persist_source(run_id: str, url: str, title: str = "", quality_score: float = 0.0, domain: str = "", subtask_id: str = "") -> None:
    await _write_async(lambda conn: conn.execute(
        "INSERT INTO sources (run_id, url, title, quality_score, domain, subtask_id) VALUES (?, ?, ?, ?, ?, ?)",
        (run_id, url, title, quality_score, domain, subtask_id),
    ))


async def persist_subagent_report(run_id: str, subtask_id: str, content: str, sources_count: int = 0, evidence_count: int = 0) -> None:
    await _write_async(lambda conn: conn.execute(
        "INSERT INTO subagent_reports (run_id, subtask_id, content, sources_count, evidence_count, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (run_id, subtask_id, content, sources_count, evidence_count, int(time.time())),
    ))


async def delete_run(run_id: str) -> bool:
    report_path: str | None = None
    deleted: bool = False

    def _delete(conn):
        nonlocal report_path, deleted
        row = conn.execute("SELECT report_path FROM runs WHERE run_id=?", (run_id,)).fetchone()
        report_path = row["report_path"] if row else None
        conn.execute("DELETE FROM checkpoints WHERE run_id=?", (run_id,))
        conn.execute("DELETE FROM sources WHERE run_id=?", (run_id,))
        conn.execute("DELETE FROM subagent_reports WHERE run_id=?", (run_id,))
        cursor = conn.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
        deleted = cursor.rowcount > 0

    await _write_async(_delete)

    if report_path:
        try:
            await asyncio.to_thread(Path(report_path).unlink, missing_ok=True)
        except OSError:
            pass

    return deleted


def get_run_history(limit: int = 20) -> list[dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT run_id, query, status, provider, model, total_sources, total_reports, iterations, started_at, completed_at, report_path FROM runs ORDER BY started_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_run_by_id(run_id: str) -> dict[str, Any] | None:
    """Get a single run's metadata from the runs table."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT run_id, query, status, provider, model, total_sources, total_reports, iterations, started_at, completed_at, report_path FROM runs WHERE run_id=?",
        (run_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_latest_checkpoint(run_id: str) -> dict[str, Any] | None:
    """Get the most recent checkpoint for a run (phase + state_json)."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT phase, state FROM checkpoints WHERE run_id=? ORDER BY created_at DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    result = dict(row)
    try:
        result["state"] = json.loads(result["state"])
    except (json.JSONDecodeError, TypeError):
        pass
    return result


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


# ── collections & documents ─────────────────────────────────────

async def persist_collection(collection_id: str, name: str, description: str = "") -> None:
    await _write_async(lambda conn: conn.execute(
        "INSERT OR REPLACE INTO collections (id, name, description, created_at) VALUES (?, ?, ?, ?)",
        (collection_id, name, description, int(time.time())),
    ))


async def delete_collection_db(collection_id: str) -> None:
    await _write_async(lambda conn: conn.execute(
        "DELETE FROM collections WHERE id=?", (collection_id,)
    ))


async def list_collections_db() -> list[dict[str, Any]]:
    def _read(conn):
        rows = conn.execute(
            """
            SELECT c.id, c.name, c.description, c.status, c.created_at,
                   COUNT(d.id) as doc_count
            FROM collections c
            LEFT JOIN documents d ON d.collection_id = c.id
            GROUP BY c.id
            ORDER BY c.created_at DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]
    return await _read_async(_read)


async def get_collection_db(collection_id: str) -> dict[str, Any] | None:
    def _read(conn):
        row = conn.execute(
            """
            SELECT c.id, c.name, c.description, c.status, c.created_at,
                   COUNT(d.id) as doc_count
            FROM collections c
            LEFT JOIN documents d ON d.collection_id = c.id
            WHERE c.id = ?
            GROUP BY c.id
            """,
            (collection_id,),
        ).fetchone()
        return dict(row) if row else None
    return await _read_async(_read)


async def persist_document(
    doc_id: str,
    collection_id: str,
    name: str,
    file_path: str,
    file_type: str,
    category: str = "",
    tags: str = "[]",
    page_count: int = 0,
    chunk_count: int = 0,
    status: str = "indexed",
    error_msg: str = "",
) -> None:
    await _write_async(lambda conn: conn.execute(
        """
        INSERT INTO documents (id, collection_id, name, file_path, file_type, category, tags, page_count, chunk_count, status, error_msg, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (doc_id, collection_id, name, file_path, file_type, category, tags, page_count, chunk_count, status, error_msg, int(time.time())),
    ))


async def delete_document_db(collection_id: str, doc_id: str) -> None:
    await _write_async(lambda conn: conn.execute(
        "DELETE FROM documents WHERE id=? AND collection_id=?",
        (doc_id, collection_id),
    ))


async def update_document_status(
    doc_id: str,
    status: str,
    chunk_count: int = 0,
    page_count: int = 0,
    error_msg: str = "",
) -> None:
    await _write_async(lambda conn: conn.execute(
        """
        UPDATE documents
        SET status=?, chunk_count=?, page_count=?, error_msg=?
        WHERE id=?
        """,
        (status, chunk_count, page_count, error_msg, doc_id),
    ))


async def get_document_db(doc_id: str) -> dict[str, Any] | None:
    def _read(conn):
        row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return dict(row) if row else None
    return await _read_async(_read)


async def list_documents_db(collection_id: str) -> list[dict[str, Any]]:
    def _read(conn):
        rows = conn.execute(
            "SELECT * FROM documents WHERE collection_id=? ORDER BY created_at DESC",
            (collection_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    return await _read_async(_read)


# ── tracing ─────────────────────────────────────────────────────

async def persist_llm_call(
    run_id: str,
    call_id: str,
    role: str,
    provider: str,
    model: str,
    messages: str,
    response: str,
    latency_ms: int = 0,
    retry_attempt: int = 0,
    error: str = "",
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> None:
    await _write_async(lambda conn: conn.execute(
        """
        INSERT INTO llm_calls
        (run_id, call_id, role, provider, model, temperature, max_tokens, messages, response, latency_ms, retry_attempt, error, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, call_id, role, provider, model, temperature, max_tokens, messages, response, latency_ms, retry_attempt, error, int(time.time())),
    ))


async def persist_trace_log(
    run_id: str,
    phase: str,
    event_type: str,
    message: str,
    details: dict[str, Any] | None = None,
    level: str = "info",
    parent_id: int | None = None,
) -> int:
    def _write(conn) -> int:
        cursor = conn.execute(
            """
            INSERT INTO trace_logs
            (run_id, phase, level, event_type, message, details, parent_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, phase, level, event_type, message, json.dumps(details) if details else None, parent_id, int(time.time())),
        )
        return cursor.lastrowid

    return await _write_async(_write)


def get_run_logs(
    run_id: str,
    phase: str | None = None,
    level: str | None = None,
    event_type: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    conn = _get_conn()
    query = "SELECT * FROM trace_logs WHERE run_id=?"
    params: list[Any] = [run_id]
    if phase:
        query += " AND phase=?"
        params.append(phase)
    if level:
        query += " AND level=?"
        params.append(level)
    if event_type:
        query += " AND event_type=?"
        params.append(event_type)
    query += " ORDER BY created_at ASC, id ASC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_run_llm_calls(
    run_id: str,
    role: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    conn = _get_conn()
    query = "SELECT * FROM llm_calls WHERE run_id=?"
    params: list[Any] = [run_id]
    if role:
        query += " AND role=?"
        params.append(role)
    query += " ORDER BY created_at ASC, id ASC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_run_timeline(run_id: str, limit: int = 1000) -> list[dict[str, Any]]:
    """Merge trace_logs and llm_calls into a single chronological timeline."""
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT
            'trace' as source,
            id,
            phase,
            event_type as type,
            level,
            message,
            details,
            NULL as role,
            NULL as model,
            NULL as latency_ms,
            created_at
        FROM trace_logs WHERE run_id=?
        UNION ALL
        SELECT
            'llm' as source,
            id,
            'llm' as phase,
            'llm_call' as type,
            'info' as level,
            'LLM call [' || role || '] ' || CASE WHEN error IS NULL OR error='' THEN 'success' ELSE 'error' END as message,
            json_object('role', role, 'model', model, 'provider', provider, 'temperature', temperature, 'max_tokens', max_tokens, 'retry_attempt', retry_attempt, 'error', error, 'messages_preview', substr(messages, 1, 500), 'response_preview', substr(response, 1, 500)) as details,
            role,
            model,
            latency_ms,
            created_at
        FROM llm_calls WHERE run_id=?
        ORDER BY created_at ASC, id ASC
        LIMIT ?
        """,
        (run_id, run_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

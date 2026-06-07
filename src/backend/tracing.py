"""Structured tracing for research runs.

Uses contextvars to propagate run_id implicitly through async call stacks,
so business logic functions don't need to pass run_id explicitly.

Log levels:
- debug : detailed traces (full LLM messages, RAG scores, state diffs)
- info  : normal traces (node enter/exit, decisions, LLM call summaries)
- warning: warnings (fallbacks, retries, low-quality results)
- error : errors (exceptions, failures)

The configured log_level in AppConfig controls which traces are persisted.
Default is "info", meaning debug-level details are skipped.
"""

import contextvars
import json
import uuid
from typing import Any

from .config import get_config
from .persistence import persist_llm_call, persist_trace_log

_LEVEL_ORDER = {"debug": 0, "info": 1, "warning": 2, "error": 3}

current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_run_id", default=None
)

current_phase: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_phase", default=None
)


def _effective_level() -> int:
    cfg = get_config()
    return _LEVEL_ORDER.get(getattr(cfg, "log_level", "info").lower(), 1)


def _should_log(level: str) -> bool:
    """Return True if the given level should be persisted."""
    return _LEVEL_ORDER.get(level.lower(), 1) >= _effective_level()


def _truncate_for_level(
    messages: list[dict[str, str]], response: str | None, level: str
) -> tuple[str, str]:
    """In non-debug mode, truncate messages and response to save space."""
    if level == "debug":
        return json.dumps(messages, ensure_ascii=False), response or ""

    # info mode: keep only message roles + first 120 chars of content
    summary = []
    for m in messages:
        content = m.get("content", "")
        preview = content[:120] + "..." if len(content) > 120 else content
        summary.append({"role": m.get("role", ""), "content_preview": preview})

    resp_preview = (response or "")[:200] + "..." if len(response or "") > 200 else (response or "")
    return json.dumps(summary, ensure_ascii=False), resp_preview


async def trace(
    phase: str,
    event_type: str,
    message: str,
    details: dict[str, Any] | None = None,
    level: str = "info",
    parent_id: int | None = None,
) -> int | None:
    """Write a trace log entry for the current run_id (from contextvar).

    Returns the inserted row id, or None if no run_id is set or level is filtered.
    """
    run_id = current_run_id.get()
    if not run_id:
        return None
    if not _should_log(level):
        return None

    # In non-debug mode, strip large details to keep traces lightweight
    stored_details = details
    if level == "debug" and _effective_level() > 0:
        # If caller explicitly marked as debug but config is info, skip entirely
        return None

    return await persist_trace_log(
        run_id=run_id,
        phase=phase,
        event_type=event_type,
        message=message,
        details=stored_details,
        level=level,
        parent_id=parent_id,
    )


async def trace_llm_call(
    role: str,
    messages: list[dict[str, str]],
    provider: str,
    model: str,
    response: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    latency_ms: int = 0,
    retry_attempt: int = 0,
    error: str | None = None,
) -> None:
    """Write an LLM call record for the current run_id.

    In info mode only a summary (roles + previews) is stored.
    In debug mode the full messages and response are stored.
    """
    run_id = current_run_id.get()
    if not run_id:
        return

    level = "error" if error else "info"
    if not _should_log(level):
        return

    is_debug = _effective_level() == 0
    if not is_debug:
        # Store as info-level summary (truncated)
        msgs, resp = _truncate_for_level(messages, response, "info")
    else:
        msgs, resp = json.dumps(messages, ensure_ascii=False), response or ""

    call_id = str(uuid.uuid4())
    await persist_llm_call(
        run_id=run_id,
        call_id=call_id,
        role=role,
        phase=current_phase.get() or "llm",
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=msgs,
        response=resp,
        latency_ms=latency_ms,
        retry_attempt=retry_attempt,
        error=error,
    )

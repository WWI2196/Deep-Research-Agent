"""Supabase persistence layer for checkpoints and artifacts."""

import os
import time
from typing import Any, Dict, Optional

from supabase import create_client


def _client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def persist_state(state: dict, phase: str) -> None:
    client = _client()
    if not client:
        return
    try:
        client.table("deep_research_checkpoints").insert({
            "run_id": state.get("run_id"),
            "phase": phase,
            "state": state,
            "created_at": int(time.time()),
        }).execute()
    except Exception:
        try:
            client.table("deep_research_runs").upsert({
                "run_id": state.get("run_id"),
                "phase": phase,
                "state": state,
                "updated_at": int(time.time()),
            }, on_conflict="run_id").execute()
        except Exception:
            pass


def persist_artifact(
    run_id: str,
    artifact_type: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    client = _client()
    if not client:
        return
    try:
        client.table("deep_research_artifacts").insert({
            "run_id": run_id,
            "artifact_type": artifact_type,
            "content": content,
            "metadata": metadata or {},
            "created_at": int(time.time()),
        }).execute()
    except Exception:
        pass

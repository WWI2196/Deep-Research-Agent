"""FastAPI server — serves API and static frontend."""

import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import get_config, reload_config, save_config
from .document_store import DocumentStore
from .export import export_markdown
from .graph import build_and_run_graph
from .models import (
    CollectionCreateRequest,
    CollectionUpdateRequest,
    ConfigUpdateRequest,
    ResearchRequest,
    ResearchResponse,
)
from .persistence import (
    delete_run,
    get_latest_checkpoint,
    get_report_content,
    get_run_by_id,
    get_run_history,
    get_run_logs,
    get_run_llm_calls,
    get_run_report,
    get_run_timeline,
    init_db,
    persist_run,
    update_run_status,
)

load_dotenv()

app = FastAPI(title="Deep Research Agent", version="1.0.0")
logger = logging.getLogger(__name__)

active_runs: dict[str, dict] = {}
_cancel_flags: dict[str, asyncio.Event] = {}
_doc_store = DocumentStore()


def serialize_event(event_type: str, data: dict) -> str:
    payload = json.dumps({"type": event_type, **data}, default=str)
    return f"data: {payload}\n\n"


def _get_error_hint(error_msg: str) -> str:
    lower = error_msg.lower()
    if "402" in lower or "payment" in lower:
        return "API credits depleted. Check your billing at the provider's dashboard."
    if "404" in lower or "not found" in lower:
        return "Model not found or temporarily unavailable."
    if "401" in lower or "unauthorized" in lower:
        return "Invalid API key. Check your settings."
    if "403" in lower:
        return "Access denied. Your API key may lack permissions for this model."
    if "rate" in lower or "429" in lower:
        return "Rate limited. Will retry automatically."
    return ""


async def _process_events(event_queue: asyncio.Queue, timeout: float = 0.05):
    """Drain events from the queue until empty."""
    events = []
    while True:
        try:
            event = event_queue.get_nowait()
            events.append(event)
        except asyncio.QueueEmpty:
            break
    return events


@app.on_event("startup")
async def startup():
    init_db()


async def _run_research(state: dict[str, Any], run_id: str) -> None:
    """Background task wrapper that executes the research graph."""
    try:
        async def _noop_event(evt: dict[str, Any]) -> None:
            pass

        final_state = await build_and_run_graph(
            state, _noop_event, cancel_event=_cancel_flags.get(run_id),
        )

        final_content = final_state.get("cited_report") or final_state.get("report", "")
        report_path = ""
        if final_content:
            report_path = export_markdown(final_content)

        source_count = len(final_state.get("sources", []))
        report_count = len(final_state.get("subagent_reports", []))
        iterations = final_state.get("iteration_count", 0)

        await update_run_status(
            run_id, "completed",
            total_sources=source_count,
            total_reports=report_count,
            iterations=iterations,
            report_path=report_path,
        )
    except asyncio.CancelledError:
        await update_run_status(run_id, "cancelled")
        raise
    except Exception as e:
        logger.exception("Research task failed: %s", e)
        await update_run_status(run_id, "failed")
    finally:
        _cancel_flags.pop(run_id, None)
        active_runs.pop(run_id, None)


@app.post("/api/research")
async def start_research(request: ResearchRequest) -> ResearchResponse:
    run_id = os.urandom(8).hex()
    _cancel_flags[run_id] = asyncio.Event()
    cfg = get_config()

    state = {
        "user_query": request.query,
        "run_id": run_id,
        "events": [],
        "errors": [],
        "max_iterations": request.max_iterations or 2,
        "quality_threshold": request.quality_threshold or cfg.quality_threshold,
        "context_compress_retries": request.context_compress_retries or cfg.context_compress_retries,
        "keep_tool_results": request.keep_tool_results or cfg.keep_tool_results,
        "document_collections": request.document_collections or [],
        "output_language": request.output_language or cfg.output_language,
    }

    task = asyncio.create_task(_run_research(state, run_id))
    active_runs[run_id] = {
        "query": request.query,
        "status": "running",
        "task": task,
    }

    await persist_run(run_id, request.query, cfg.base_url, cfg.default_model)
    return ResearchResponse(run_id=run_id, status="started")


@app.post("/api/research/stream", include_in_schema=False)
async def stream_research(request: ResearchRequest):
    run_id = os.urandom(8).hex()
    _cancel_flags[run_id] = asyncio.Event()
    cfg = get_config()

    # Register in active_runs so history page can query / cancel it
    active_runs[run_id] = {
        "query": request.query,
        "status": "running",
        "task": None,
    }

    async def generator():
        event_queue: asyncio.Queue = asyncio.Queue()

        async def on_event(event: dict):
            await event_queue.put(event)

        yield serialize_event("phase-update", {
            "phase": "init",
            "message": f"Starting research (model: {cfg.default_model})",
            "run_id": run_id,
            "model": cfg.default_model,
        })

        try:
            state = {
                "user_query": request.query,
                "run_id": run_id,
                "events": [],
                "errors": [],
                "max_iterations": request.max_iterations or 2,
                "quality_threshold": request.quality_threshold or cfg.quality_threshold,
                "context_compress_retries": request.context_compress_retries or cfg.context_compress_retries,
                "keep_tool_results": request.keep_tool_results or cfg.keep_tool_results,
                "document_collections": request.document_collections or [],
                "output_language": request.output_language or cfg.output_language,
            }

            graph_task = asyncio.create_task(
                build_and_run_graph(state, on_event, cancel_event=_cancel_flags.get(run_id))
            )
            active_runs[run_id]["task"] = graph_task

            # Clean up resources when the task finishes (regardless of client disconnect)
            def _cleanup_task(_fut):
                _cancel_flags.pop(run_id, None)
                active_runs.pop(run_id, None)

            graph_task.add_done_callback(_cleanup_task)

            while not graph_task.done():
                done, _ = await asyncio.wait([graph_task], timeout=0.1)
                events = await _process_events(event_queue)
                for evt in events:
                    for sse in _translate_event(evt):
                        yield sse

            # Drain remaining events
            events = await _process_events(event_queue)
            for evt in events:
                for sse in _translate_event(evt):
                    yield sse

            final_state = graph_task.result()
            final_content = final_state.get("cited_report") or final_state.get("report", "")
            report_path = ""
            if final_content:
                yield serialize_event("final-result", {"content": final_content})
                report_path = export_markdown(final_content)

            source_count = len(final_state.get("sources", []))
            report_count = len(final_state.get("subagent_reports", []))
            iterations = final_state.get("iteration_count", 0)

            await update_run_status(
                run_id, "completed",
                total_sources=source_count,
                total_reports=report_count,
                iterations=iterations,
                report_path=report_path,
            )

            yield serialize_event("complete", {
                "message": "Research complete",
                "run_id": run_id,
                "total_sources": source_count,
                "total_reports": report_count,
                "iterations": iterations,
                "model": cfg.default_model,
            })

        except asyncio.CancelledError:
            # Client disconnected — stop streaming, but do NOT cancel the research task.
            # graph_task continues running in background; _cleanup_task handles cleanup.
            return

        except Exception as e:
            await update_run_status(run_id, "failed")
            yield serialize_event("error", {
                "error": str(e),
                "phase": "pipeline",
                "hint": _get_error_hint(str(e)),
            })

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/research/{run_id}/cancel")
async def cancel_research(run_id: str):
    flag = _cancel_flags.get(run_id)
    if flag:
        flag.set()
        entry = active_runs.get(run_id)
        if entry and entry.get("task"):
            entry["task"].cancel()
        return {"status": "cancelled", "run_id": run_id}
    raise HTTPException(status_code=404, detail="Research run not found")


@app.get("/api/research/history")
async def history(limit: int = 20):
    return {"history": get_run_history(limit)}


@app.get("/api/research/{run_id}/status")
async def get_research_status(run_id: str):
    run = get_run_by_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Research run not found")

    checkpoint = get_latest_checkpoint(run_id)
    phase = checkpoint.get("phase", "") if checkpoint else ""
    state = checkpoint.get("state", {}) if checkpoint else {}

    # Approximate progress mapping based on phase
    progress_percent = 0
    phase_progress = {
        "init": 5, "plan": 15, "split": 25,
        "subagents": 60, "reflection": 75, "synthesize": 90, "cite": 95,
    }
    for key, pct in phase_progress.items():
        if phase.startswith(key):
            progress_percent = pct
            break

    active = run_id in active_runs

    return {
        "run_id": run_id,
        "status": run.get("status", "unknown"),
        "phase": phase,
        "progress_percent": progress_percent,
        "iteration": state.get("iteration_count", 0),
        "total_reports": len(state.get("subagent_reports", [])),
        "total_sources": len(state.get("sources", [])),
        "query": run.get("query", ""),
        "active": active,
        "started_at": run.get("started_at"),
        "completed_at": run.get("completed_at"),
    }


@app.delete("/api/research/{run_id}")
async def delete_research(run_id: str):
    deleted = await delete_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Research run not found")
    return {"status": "deleted", "run_id": run_id}


@app.get("/api/research/{run_id}/report")
async def get_report(run_id: str):
    report = get_run_report(run_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    report["content"] = get_report_content(run_id)
    return report


@app.get("/api/config")
async def get_app_config():
    cfg = get_config()
    api_key_masked = ""
    if cfg.api_key:
        api_key_masked = cfg.api_key[:4] + "****" + cfg.api_key[-4:] if len(cfg.api_key) > 8 else "****"
    return {
        "base_url": cfg.base_url,
        "api_key": api_key_masked,
        "default_model": cfg.default_model,
        "quality_threshold": cfg.quality_threshold,
        "context_compress_retries": cfg.context_compress_retries,
        "keep_tool_results": cfg.keep_tool_results,
        "log_level": cfg.log_level,
        "roles": cfg.to_dict().get("roles", {}),
    }


@app.post("/api/config")
async def update_config(request: ConfigUpdateRequest):
    cfg = reload_config()
    if request.base_url is not None:
        cfg.base_url = request.base_url
    if request.api_key is not None:
        cfg.api_key = request.api_key
    if request.default_model is not None:
        cfg.default_model = request.default_model
    if request.quality_threshold is not None:
        cfg.quality_threshold = request.quality_threshold
    if request.context_compress_retries is not None:
        cfg.context_compress_retries = request.context_compress_retries
    if request.keep_tool_results is not None:
        cfg.keep_tool_results = request.keep_tool_results
    if request.log_level is not None:
        cfg.log_level = request.log_level
    if request.roles:
        from .config import RoleConfig
        for role_name, role_data in request.roles.items():
            cfg.roles[role_name] = RoleConfig(
                model=role_data.get("model", cfg.default_model),
            )
    save_config(cfg)
    from . import llm
    llm.invalidate_client_cache()
    return {"status": "saved"}


@app.get("/api/health")
async def health():
    cfg = get_config()
    return {
        "status": "healthy",
        "version": "1.0.0",
        "base_url": cfg.base_url,
        "model": cfg.default_model,
    }


@app.get("/api/models")
async def list_models():
    return {"providers": [], "details": {}}


# ── Tracing / Logs ──────────────────────────────────────────────

@app.get("/api/research/{run_id}/logs")
async def get_logs(run_id: str, phase: str = "", level: str = "", event_type: str = "", limit: int = 2000):
    logs = get_run_logs(
        run_id,
        phase=phase or None,
        level=level or None,
        event_type=event_type or None,
        limit=limit,
    )
    return {"run_id": run_id, "logs": logs}


@app.get("/api/research/{run_id}/llm-calls")
async def get_llm_calls(run_id: str, role: str = "", limit: int = 2000):
    calls = get_run_llm_calls(run_id, role=role or None, limit=limit)
    return {"run_id": run_id, "calls": calls}


@app.get("/api/research/{run_id}/timeline")
async def get_timeline(run_id: str, limit: int = 3000):
    items = get_run_timeline(run_id, limit=limit)
    return {"run_id": run_id, "items": items}


# ── Document collections ────────────────────────────────────────

@app.get("/api/collections")
async def get_collections():
    collections = await _doc_store.list_collections()
    return {"collections": collections}


@app.post("/api/collections")
async def create_collection(request: CollectionCreateRequest):
    collection = await _doc_store.create_collection(request.name, request.description)
    return collection


@app.delete("/api/collections/{collection_id}")
async def delete_collection(collection_id: str):
    success = await _doc_store.delete_collection(collection_id)
    if not success:
        raise HTTPException(status_code=404, detail="Collection not found")
    return {"status": "deleted", "id": collection_id}


@app.patch("/api/collections/{collection_id}")
async def update_collection(collection_id: str, request: CollectionUpdateRequest):
    # TODO: implement name/description update
    return {"status": "not_implemented", "id": collection_id}


@app.get("/api/collections/{collection_id}/documents")
async def get_documents(collection_id: str):
    documents = await _doc_store.list_documents(collection_id)
    return {"documents": documents}


@app.post("/api/collections/{collection_id}/documents")
async def upload_document(collection_id: str, file: UploadFile = File(...)):
    import tempfile

    suffix = Path(file.filename or "upload").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = await _doc_store.add_document(
            collection_id=collection_id,
            file_path=tmp_path,
            name=file.filename or "unnamed",
        )
        if not result["success"]:
            raise HTTPException(status_code=422, detail=result.get("error", "Parse failed"))
        return result
    finally:
        tmp_path.unlink(missing_ok=True)


@app.delete("/api/collections/{collection_id}/documents/{doc_id}")
async def delete_document(collection_id: str, doc_id: str):
    success = await _doc_store.delete_document(collection_id, doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted", "id": doc_id}


@app.get("/api/collections/{collection_id}/documents/{doc_id}/download")
async def download_document(collection_id: str, doc_id: str):
    from fastapi.responses import FileResponse

    doc_dir = Path.home() / ".deep-research" / "docs" / collection_id
    if not doc_dir.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    for p in doc_dir.iterdir():
        if p.stem == doc_id:
            return FileResponse(str(p), filename=p.name)

    raise HTTPException(status_code=404, detail="Document not found")


@app.post("/api/collections/{collection_id}/search")
async def search_collection(collection_id: str, body: dict):
    query = body.get("query", "")
    top_k = body.get("top_k", 10)
    category = body.get("category")
    if not query:
        raise HTTPException(status_code=422, detail="query is required")
    results = await _doc_store.query([collection_id], query, top_k=top_k, category_filter=category)
    return {"results": results}


@app.post("/api/collections/{collection_id}/reindex")
async def reindex_collection(collection_id: str):
    result = await _doc_store.reindex_collection(collection_id)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Reindex failed"))
    return result


@app.post("/api/collections/{collection_id}/documents/{doc_id}/reindex")
async def reindex_document(collection_id: str, doc_id: str):
    result = await _doc_store.reindex_document(collection_id, doc_id)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result.get("error", "Document not found"))
    return result


HERE = Path(__file__).parent          # src/backend/
RENDERER_DIR = HERE.parent / "renderer"  # src/renderer/

app.mount("/", StaticFiles(directory=str(RENDERER_DIR), html=True), name="static")


def _translate_event(evt: dict) -> list[str]:
    t = evt.get("type", "")
    results: list[str] = []

    mappers: dict[str, str] = {
        "phase-update": "phase-update",
        "plan-generated": "plan-generated",
        "subtasks-created": "subtasks-created",
        "scaling-computed": "scaling-computed",
        "subagents-launch": "subagents-launch",
        "subagent-step": "subagent-step",
        "subagent-queries": "subagent-queries",
        "subagent-search": "subagent-search",
        "subagent-sources-scored": "subagent-sources-scored",
        "subagent-extract": "subagent-extract",
        "subagent-complete": "subagent-complete",
        "llm-call-start": "llm-call",
        "llm-call-end": "llm-call",
        "llm-call-error": "llm-call",
        "reflection-decision": "reflection-decision",
        "report-draft": "report-draft",
        "report-synthesized": "report-draft",
        "citations-added": "citations-added",
        "progress": "progress",
        "warning": "warning",
        "error": "error",
    }

    mapped_type = mappers.get(t, "log")
    payload = dict(evt)
    payload["type"] = mapped_type
    results.append(serialize_event(mapped_type, payload))
    return results

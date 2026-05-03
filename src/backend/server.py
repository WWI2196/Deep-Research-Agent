"""FastAPI server — serves API and static frontend."""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import get_config, reload_config, save_config
from .export import export_markdown
from .graph import build_and_run_graph
from .models import ConfigUpdateRequest, ResearchRequest, ResearchResponse
from .persistence import (
    delete_run,
    get_report_content,
    get_run_history,
    get_run_report,
    init_db,
    update_run_status,
)

load_dotenv()

app = FastAPI(title="Deep Research Agent", version="1.0.0")


active_runs: dict[str, dict] = {}
_cancel_flags: dict[str, asyncio.Event] = {}


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


@app.post("/api/research")
async def start_research(request: ResearchRequest) -> ResearchResponse:
    run_id = os.urandom(8).hex()
    active_runs[run_id] = {"query": request.query, "status": "started"}
    _cancel_flags[run_id] = asyncio.Event()
    return ResearchResponse(run_id=run_id, status="started")


@app.post("/api/research/stream")
async def stream_research(request: ResearchRequest):
    run_id = os.urandom(8).hex()
    _cancel_flags[run_id] = asyncio.Event()
    cfg = get_config()

    async def generator():
        event_queue: asyncio.Queue = asyncio.Queue()

        async def on_event(event: dict):
            await event_queue.put(event)

        yield serialize_event("phase-update", {
            "phase": "init",
            "message": f"Starting research (provider: {cfg.default_provider}, model: {cfg.default_model})",
            "run_id": run_id,
            "provider": cfg.default_provider,
            "model": cfg.default_model,
        })

        try:
            state = {
                "user_query": request.query,
                "run_id": run_id,
                "events": [],
                "errors": [],
                "max_iterations": request.max_iterations or cfg.max_iterations,
                "quality_threshold": request.quality_threshold or cfg.quality_threshold,
            }

            graph_task = asyncio.create_task(
                build_and_run_graph(state, on_event)
            )

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
                "provider": cfg.default_provider,
                "model": cfg.default_model,
            })

        except asyncio.CancelledError:
            await update_run_status(run_id, "cancelled")
            yield serialize_event("error", {"error": "Research cancelled", "phase": "pipeline"})
        except Exception as e:
            await update_run_status(run_id, "failed")
            yield serialize_event("error", {
                "error": str(e),
                "phase": "pipeline",
                "hint": _get_error_hint(str(e)),
            })
        finally:
            _cancel_flags.pop(run_id, None)
            active_runs.pop(run_id, None)

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
        return {"status": "cancelled", "run_id": run_id}
    raise HTTPException(status_code=404, detail="Research run not found")


@app.get("/api/research/history")
async def history(limit: int = 20):
    return {"history": get_run_history(limit)}


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
    visible = {name: {"type": pc.type} for name, pc in cfg.providers.items() if pc.api_key}
    return {
        "default_provider": cfg.default_provider,
        "default_model": cfg.default_model,
        "max_iterations": cfg.max_iterations,
        "quality_threshold": cfg.quality_threshold,
        "providers": visible,
        "available_providers": list(visible.keys()),
        "roles": cfg.to_dict().get("roles", {}),
    }


@app.post("/api/config")
async def update_config(request: ConfigUpdateRequest):
    cfg = reload_config()
    if request.default_provider is not None:
        cfg.default_provider = request.default_provider
    if request.default_model is not None:
        cfg.default_model = request.default_model
    if request.max_iterations is not None:
        cfg.max_iterations = request.max_iterations
    if request.quality_threshold is not None:
        cfg.quality_threshold = request.quality_threshold
    if request.roles:
        from .config import RoleConfig
        for role_name, role_data in request.roles.items():
            cfg.roles[role_name] = RoleConfig(
                provider=role_data.get("provider", cfg.default_provider),
                model=role_data.get("model", cfg.default_model),
            )
    save_config(cfg)
    return {"status": "saved"}


@app.get("/api/health")
async def health():
    cfg = get_config()
    return {
        "status": "healthy",
        "version": "1.0.0",
        "provider": cfg.default_provider,
        "model": cfg.default_model,
    }


@app.get("/api/models")
async def list_models():
    cfg = get_config()
    visible = {name: {"type": pc.type} for name, pc in cfg.providers.items() if pc.api_key}
    return {"providers": list(visible.keys()), "details": visible}


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

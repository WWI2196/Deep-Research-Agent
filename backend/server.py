"""FastAPI server with SSE for real-time research streaming.

Routes every LLM call through the pluggable provider abstraction
(Gemini / OpenAI / Claude / HuggingFace).
"""

import asyncio
import json
import os
import queue
import uuid
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .events import add_listener, remove_listener
from .graph import build_graph
from .models import ResearchState
from .config import get_config, reload_config
from .providers import list_providers, get_provider

load_dotenv()

app = FastAPI(
    title="Deep Research Agent API",
    description="Multi-provider AI research agent with real-time streaming",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        os.environ.get("FRONTEND_URL", "http://localhost:3000"),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── in-memory store ─────────────────────────────────────────────────
active_runs: dict = {}


# ── request / response models ───────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str
    max_iterations: Optional[int] = 2
    quality_threshold: Optional[float] = 0.7
    model: Optional[str] = None
    provider: Optional[str] = None


class ResearchResponse(BaseModel):
    run_id: str
    status: str
    report: Optional[str] = None


# ── helpers ──────────────────────────────────────────────────────────

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
        return "Invalid API key. Check your .env file."
    if "403" in lower:
        return "Access denied. Your API key may lack permissions for this model."
    if "firecrawl" in lower:
        return "Firecrawl API error. Check FIRECRAWL_API_KEY."
    if "rate" in lower or "429" in lower:
        return "Rate limited. The system will retry automatically."
    return ""


def _translate_event(evt: dict) -> list:
    """Translate an event-bus event into SSE events."""
    t = evt.get("type", "")
    results = []

    if t == "phase-start":
        results.append(serialize_event("phase-update", {
            "phase": evt.get("phase", ""),
            "message": evt.get("message", ""),
        }))
    elif t == "plan-generated":
        results.append(serialize_event("thinking", {
            "phase": "plan",
            "message": evt.get("plan_preview", ""),
            "plan_length": evt.get("plan_length", 0),
        }))
    elif t == "subtasks-created":
        results.append(serialize_event("subtasks", {
            "count": evt.get("count", 0),
            "titles": [s.get("title", "") for s in evt.get("subtasks", [])],
            "subtasks": evt.get("subtasks", []),
        }))
    elif t == "scaling-computed":
        sc = evt.get("scaling", {})
        results.append(serialize_event("scaling", {
            "complexity": sc.get("complexity", "unknown"),
            "subagent_count": sc.get("subagent_count", 0),
            "target_sources": sc.get("target_sources", 0),
            "tool_calls_per_subagent": sc.get("tool_calls_per_subagent", 0),
        }))
    elif t == "subagents-launch":
        results.append(serialize_event("subagents-launch", {
            "iteration": evt.get("iteration", 1),
            "total_agents": evt.get("total_agents", 0),
            "agent_details": evt.get("agent_details", []),
        }))
    elif t == "subagent-step":
        results.append(serialize_event("subagent-step", {
            "subtask_id": evt.get("subtask_id", ""),
            "subtask_title": evt.get("subtask_title", ""),
            "step": evt.get("step", ""),
            "message": evt.get("message", ""),
            "evidence_count": evt.get("evidence_count"),
        }))
    elif t == "subagent-queries":
        results.append(serialize_event("subagent-queries", {
            "subtask_id": evt.get("subtask_id", ""),
            "subtask_title": evt.get("subtask_title", ""),
            "queries": evt.get("queries", []),
            "count": evt.get("count", 0),
        }))
    elif t == "subagent-search":
        results.append(serialize_event("subagent-search", {
            "subtask_id": evt.get("subtask_id", ""),
            "query": evt.get("query", ""),
            "status": evt.get("status", ""),
            "results_found": evt.get("results_found"),
            "error": evt.get("error"),
        }))
    elif t == "subagent-sources-scored":
        results.append(serialize_event("subagent-sources-scored", {
            "subtask_id": evt.get("subtask_id", ""),
            "subtask_title": evt.get("subtask_title", ""),
            "total_candidates": evt.get("total_candidates", 0),
            "unique_sources": evt.get("unique_sources", 0),
            "top_urls": evt.get("top_urls", []),
            "top_scores": evt.get("top_scores", []),
        }))
    elif t == "subagent-extract":
        results.append(serialize_event("subagent-extract", {
            "subtask_id": evt.get("subtask_id", ""),
            "url": evt.get("url", ""),
            "status": evt.get("status", ""),
            "error": evt.get("error"),
        }))
    elif t == "subagent-complete":
        results.append(serialize_event("subagent-complete", {
            "subtask_id": evt.get("subtask_id", ""),
            "subtask_title": evt.get("subtask_title", ""),
            "report_length": evt.get("report_length", 0),
            "sources_count": evt.get("sources_count", 0),
            "evidence_count": evt.get("evidence_count", 0),
        }))
    elif t == "llm-call-start":
        results.append(serialize_event("llm-call", {
            "status": "started",
            "model": evt.get("model", ""),
            "provider": evt.get("provider", ""),
            "role": evt.get("role", ""),
            "attempt": evt.get("attempt", 1),
        }))
    elif t == "llm-call-end":
        results.append(serialize_event("llm-call", {
            "status": "completed",
            "model": evt.get("model", ""),
            "provider": evt.get("provider", ""),
            "role": evt.get("role", ""),
            "output_length": evt.get("output_length", 0),
        }))
    elif t == "llm-call-error":
        results.append(serialize_event("llm-call", {
            "status": "error",
            "model": evt.get("model", ""),
            "provider": evt.get("provider", ""),
            "role": evt.get("role", ""),
            "error": evt.get("error", ""),
            "attempt": evt.get("attempt", 1),
        }))
    elif t == "reflection-decision":
        results.append(serialize_event("reflection", {
            "decision": evt.get("decision", ""),
            "iteration": evt.get("iteration", 0),
            "new_subtask_count": evt.get("new_subtask_count", 0),
            "new_subtasks": evt.get("new_subtasks", []),
            "total_reports": evt.get("total_reports", 0),
            "total_sources": evt.get("total_sources", 0),
            "research_complete": evt.get("decision") in ("research-complete", "max-iterations-reached"),
        }))
    elif t == "report-synthesized":
        results.append(serialize_event("report-draft", {
            "content": evt.get("report_preview", ""),
            "report_length": evt.get("report_length", 0),
        }))
    elif t == "citations-added":
        results.append(serialize_event("citations-added", {
            "cited_report_length": evt.get("cited_report_length", 0),
        }))
    elif t == "warning":
        results.append(serialize_event("warning", {
            "phase": evt.get("phase", ""),
            "message": evt.get("message", ""),
        }))
    elif t == "progress":
        results.append(serialize_event("progress", {
            "phase": evt.get("phase", ""),
            "percent": evt.get("percent", 0),
        }))
    elif t:
        results.append(serialize_event("log", {
            "message": evt.get("message", json.dumps(evt, default=str)[:200]),
            "event_type": t,
        }))

    return results


# ── SSE generator ────────────────────────────────────────────────────

async def research_stream_generator(
    query: str,
    run_id: str,
    max_iterations: int,
    quality_threshold: float,
    model: str = None,
    provider: str = None,
):
    event_queue: queue.Queue = queue.Queue()

    def on_event(event: dict):
        event_queue.put(event)

    add_listener(on_event)

    # If model/provider are specified, temporarily update the config
    cfg = get_config()
    original_model = cfg.default_model
    original_provider = cfg.default_provider
    if provider and provider != cfg.default_provider:
        cfg.default_provider = provider
        cfg.default_model = model or ""
    elif model and model != cfg.default_model:
        cfg.default_model = model

    yield serialize_event("phase-update", {
        "phase": "init",
        "message": f"Starting research (provider: {cfg.default_provider}, model: {cfg.default_model})",
        "run_id": run_id,
        "provider": cfg.default_provider,
        "model": cfg.default_model,
    })

    try:
        graph = build_graph().compile()

        state: ResearchState = {
            "user_query": query,
            "run_id": run_id,
            "events": [],
            "errors": [],
            "max_iterations": max_iterations,
            "quality_threshold": quality_threshold,
        }

        loop = asyncio.get_event_loop()
        graph_future = loop.run_in_executor(None, graph.invoke, state)

        while True:
            if graph_future.done():
                while not event_queue.empty():
                    try:
                        evt = event_queue.get_nowait()
                        for sse in _translate_event(evt):
                            yield sse
                    except queue.Empty:
                        break
                break

            drained = 0
            while drained < 50:
                try:
                    evt = event_queue.get_nowait()
                    for sse in _translate_event(evt):
                        yield sse
                    drained += 1
                except queue.Empty:
                    break

            await asyncio.sleep(0.1)

        try:
            final_state = graph_future.result()
            final_content = final_state.get("cited_report") or final_state.get("report", "")

            if final_content:
                yield serialize_event("final-result", {"content": final_content})

            yield serialize_event("complete", {
                "message": "Research complete!",
                "run_id": run_id,
                "total_sources": len(final_state.get("sources", [])),
                "total_reports": len(final_state.get("subagent_reports", [])),
                "iterations": final_state.get("iteration_count", 0),
                "provider": cfg.default_provider,
                "model": cfg.default_model,
            })
        except Exception as e:
            yield serialize_event("error", {
                "error": str(e),
                "phase": "pipeline",
                "hint": _get_error_hint(str(e)),
            })

    except Exception as e:
        yield serialize_event("error", {
            "error": str(e),
            "phase": "unknown",
            "hint": _get_error_hint(str(e)),
        })

    finally:
        remove_listener(on_event)
        active_runs.pop(run_id, None)
        # Restore original config
        cfg.default_model = original_model
        cfg.default_provider = original_provider


# ── routes ───────────────────────────────────────────────────────────

@app.post("/api/research")
async def start_research(request: ResearchRequest):
    run_id = str(uuid.uuid4())
    active_runs[run_id] = {
        "query": request.query,
        "status": "started",
        "max_iterations": request.max_iterations,
        "quality_threshold": request.quality_threshold,
    }
    return {"run_id": run_id, "status": "started"}


@app.get("/api/research/{run_id}/stream")
async def stream_research(run_id: str):
    run = active_runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Research run not found")
    return StreamingResponse(
        research_stream_generator(
            query=run["query"],
            run_id=run_id,
            max_iterations=run.get("max_iterations", 2),
            quality_threshold=run.get("quality_threshold", 0.7),
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/api/research/stream")
async def stream_research_direct(request: ResearchRequest):
    run_id = str(uuid.uuid4())
    active_runs[run_id] = {
        "query": request.query,
        "status": "running",
        "max_iterations": request.max_iterations,
        "quality_threshold": request.quality_threshold,
    }
    return StreamingResponse(
        research_stream_generator(
            query=request.query,
            run_id=run_id,
            max_iterations=request.max_iterations or 2,
            quality_threshold=request.quality_threshold or 0.7,
            model=request.model,
            provider=request.provider,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/api/config")
async def get_app_config():
    """Return current provider configuration (no secrets)."""
    from .config import AVAILABLE_MODELS
    cfg = get_config()
    return {
        "default_provider": cfg.default_provider,
        "default_model": cfg.default_model,
        "max_iterations": cfg.max_iterations,
        "quality_threshold": cfg.quality_threshold,
        "available_providers": list_providers(),
        "available_models": AVAILABLE_MODELS,
        "role_overrides": {
            role: {"provider": rc.provider, "model": rc.model}
            for role, rc in cfg.roles.items()
        },
    }


@app.get("/api/health")
async def health_check():
    cfg = get_config()
    return {
        "status": "healthy",
        "version": "3.0.0",
        "provider": cfg.default_provider,
        "model": cfg.default_model,
        "env_check": {
            "gemini_key": bool(os.environ.get("GEMINI_API_KEY")),
            "openai_key": bool(os.environ.get("OPENAI_API_KEY")),
            "anthropic_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "hf_token": bool(os.environ.get("HF_TOKEN")),
            "firecrawl_key": bool(os.environ.get("FIRECRAWL_API_KEY")),
            "supabase": bool(os.environ.get("SUPABASE_URL")),
            "langsmith": bool(os.environ.get("LANGSMITH_API_KEY")),
        },
    }

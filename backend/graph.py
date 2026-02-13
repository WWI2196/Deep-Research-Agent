"""LangGraph state graph for the research pipeline."""

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict, List

from langgraph.graph import END, StateGraph
from langsmith import traceable

from .agents import (
    add_citations,
    compute_scaling,
    generate_research_plan,
    run_subagent,
    split_into_subtasks,
    synthesize_report,
)
from .config import get_config
from .events import emit
from .models import ResearchState
from .persistence import persist_artifact, persist_state
from .prompts import REFLECTION_SYSTEM

logger = logging.getLogger(__name__)


async def run_subagents_parallel(
    user_query: str,
    research_plan: str,
    subtasks: List[Dict[str, Any]],
    tool_calls_per_subagent: int,
) -> Dict[str, Any]:
    tasks = [
        asyncio.to_thread(run_subagent, user_query, research_plan, st, tool_calls_per_subagent)
        for st in subtasks
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    reports, sources, successful = [], [], []
    for r in results:
        if isinstance(r, Exception):
            logger.warning(f"Subagent failed: {r}")
            continue
        reports.append(r["report"])
        sources.extend(r.get("sources", []))
        successful.append(r)

    seen = set()
    unique = []
    for s in sources:
        u = s.get("url")
        if u and u not in seen:
            seen.add(u)
            unique.append(s)

    return {
        "reports": reports,
        "sources": unique,
        "raw": successful,
        "success_count": len(successful),
        "total_count": len(subtasks),
    }


def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)
    cfg = get_config()

    # ── nodes ────────────────────────────────────────────────

    # Progress tracking: each phase has a weight for %
    PHASE_WEIGHTS = {
        "init": 2, "plan": 8, "split": 5, "scale": 5,
        "subagents": 55, "reflection": 5, "synthesize": 12, "cite": 8,
    }
    total_weight = sum(PHASE_WEIGHTS.values())
    completed_weight = [0]  # mutable for closures

    def _emit_progress(phase: str, partial: float = 1.0):
        """Emit a progress event. partial=0..1 within the phase."""
        w = PHASE_WEIGHTS.get(phase, 0) * partial
        pct = min(99, int((completed_weight[0] + w) / total_weight * 100))
        emit({"type": "progress", "phase": phase, "percent": pct})

    def _finish_phase(phase: str):
        completed_weight[0] += PHASE_WEIGHTS.get(phase, 0)

    @traceable(name="init_state")
    def init_state(state: ResearchState) -> ResearchState:
        state["run_id"] = state.get("run_id") or str(uuid.uuid4())
        state["events"] = state.get("events", [])
        state["errors"] = state.get("errors", [])
        state["subagent_reports"] = state.get("subagent_reports", [])
        state["sources"] = state.get("sources", [])
        state["completed_subtasks"] = state.get("completed_subtasks", [])
        state["iteration_count"] = 0
        state["max_iterations"] = state.get("max_iterations") or cfg.max_iterations
        state["research_complete"] = False
        state["quality_threshold"] = state.get("quality_threshold") or cfg.quality_threshold
        state["current_quality_score"] = 0.0
        state["memory"] = {}
        emit({
            "type": "phase-start", "phase": "init",
            "run_id": state["run_id"],
            "message": f"Research session initialised (provider: {cfg.default_provider}, model: {cfg.default_model})",
        })
        _emit_progress("init")
        _finish_phase("init")
        persist_state(state, "init")
        return state

    @traceable(name="plan_node")
    def plan_node(state: ResearchState) -> ResearchState:
        emit({"type": "phase-start", "phase": "plan", "message": "Generating research plan..."})
        _emit_progress("plan", 0.2)
        state["research_plan"] = generate_research_plan(state["user_query"])
        emit({"type": "plan-generated", "plan_preview": state["research_plan"], "plan_length": len(state["research_plan"])})
        _emit_progress("plan")
        _finish_phase("plan")
        persist_state(state, "plan")
        return state

    @traceable(name="split_node")
    def split_node(state: ResearchState) -> ResearchState:
        emit({"type": "phase-start", "phase": "split", "message": "Breaking plan into subtasks..."})
        _emit_progress("split", 0.2)
        try:
            state["subtasks"] = split_into_subtasks(state["research_plan"])
        except Exception as e:
            logger.warning(f"Splitting failed, fallback: {e}")
            emit({"type": "warning", "phase": "split", "message": f"fallback: {str(e)[:100]}"})
            state["subtasks"] = [{
                "id": "main_research", "title": state["user_query"][:80],
                "description": state["research_plan"], "objective": state["user_query"],
                "output_format": "markdown", "tool_guidance": "web search",
                "source_types": "academic, official, news", "boundaries": "",
            }]
        emit({
            "type": "subtasks-created", "count": len(state["subtasks"]),
            "subtasks": [
                {"id": s["id"], "title": s["title"], "description": s.get("description", "")[:150], "objective": s.get("objective", "")[:100]}
                for s in state["subtasks"]
            ],
        })
        _emit_progress("split")
        _finish_phase("split")
        persist_state(state, "split")
        return state

    @traceable(name="scale_node")
    def scale_node(state: ResearchState) -> ResearchState:
        emit({"type": "phase-start", "phase": "scale", "message": "Estimating complexity..."})
        _emit_progress("scale", 0.2)
        try:
            state["scaling"] = compute_scaling(state["user_query"], state["research_plan"])
        except Exception as e:
            logger.warning(f"Scaling failed, defaults: {e}")
            n = len(state.get("subtasks", []))
            state["scaling"] = {"complexity": "moderate", "subagent_count": n, "tool_calls_per_subagent": 10, "target_sources": n * 3}


        # If scaling didn't return a subagent count, use the subtasks count as a fallback
        subtasks_count = len(state.get("subtasks", []))
        if isinstance(state.get("scaling"), dict):
            state["scaling"]["recommended_subagent_count"] = state["scaling"].get("subagent_count", subtasks_count)
            state["scaling"]["subagent_count"] = subtasks_count
        emit({"type": "scaling-computed", "scaling": state["scaling"]})
        _emit_progress("scale")
        _finish_phase("scale")
        persist_state(state, "scale")
        return state

    @traceable(name="subagents_node")
    def subagents_node(state: ResearchState) -> ResearchState:
        iteration = state.get("iteration_count", 0)
        completed = set(state.get("completed_subtasks", []))
        to_run = [s for s in state.get("subtasks", []) if s["id"] not in completed]
        if not to_run:
            return state

        emit({"type": "phase-start", "phase": "subagents", "message": f"Running {len(to_run)} subagents (iteration {iteration + 1})..."})
        emit({
            "type": "subagents-launch", "iteration": iteration + 1,
            "total_agents": len(to_run),
            "agent_details": [{"id": s["id"], "title": s["title"], "description": s.get("description", "")[:200]} for s in to_run],
        })
        state["memory"]["research_plan"] = state["research_plan"]

        budget = state.get("scaling", {}).get("tool_calls_per_subagent", 10)
        _emit_progress("subagents", 0.1)
        results = asyncio.run(
            run_subagents_parallel(state["user_query"], state["research_plan"], to_run, budget)
        )

        existing_reports = state.get("subagent_reports", [])
        existing_sources = state.get("sources", [])
        state["subagent_reports"] = existing_reports + results["reports"]
        new_completed = [r["subtask_id"] for r in results.get("raw", [])]
        state["completed_subtasks"] = list(completed.union(new_completed))

        all_sources = existing_sources + results["sources"]
        seen = set()
        unique = []
        for s in all_sources:
            u = s.get("url")
            if u and u not in seen:
                seen.add(u)
                unique.append(s)
        state["sources"] = unique

        for item in results.get("raw", []):
            persist_artifact(state.get("run_id", ""), "subagent_report", item.get("report", ""), {"subtask_id": item.get("subtask_id"), "iteration": iteration})

        state["iteration_count"] = iteration + 1
        _emit_progress("subagents")
        _finish_phase("subagents")
        persist_state(state, f"subagents_iter_{iteration}")
        return state

    @traceable(name="reflection_node")
    def reflection_node(state: ResearchState) -> ResearchState:
        from .agents import _chat
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", 2)
        reports = state.get("subagent_reports", [])
        subtasks = state.get("subtasks", [])
        past = ", ".join(f"{s.get('id')}: {s.get('title')}" for s in subtasks)

        if iteration >= max_iter:
            state["research_complete"] = True
            emit({"type": "reflection-decision", "decision": "max-iterations-reached", "iteration": iteration})
            return state

        emit({"type": "phase-start", "phase": "reflection", "message": f"Reflecting (iter {iteration}/{max_iter})..."})
        _emit_progress("reflection", 0.2)
        try:
            truncated = "\n\n".join(r[:3000] for r in reports)
            response = _chat(
                role="reflection",
                messages=[{
                    "role": "user",
                    "content": REFLECTION_SYSTEM.format(
                        user_query=state["user_query"],
                        research_plan=state["research_plan"][:2000],
                        past_subtasks=past,
                        subagent_reports=truncated,
                    ) + "\n\nReturn ONLY valid JSON.",
                }],
            )
            from .agents import _extract_json
            content = response.strip()
            new_subtasks = []
            try:
                payload = json.loads(content)
                new_subtasks = payload.get("subtasks", [])
            except json.JSONDecodeError:
                ext = _extract_json(content)
                if ext:
                    try:
                        new_subtasks = json.loads(ext).get("subtasks", [])
                    except Exception:
                        pass

            if new_subtasks:
                emit({
                    "type": "reflection-decision", "decision": "gaps-found",
                    "new_subtask_count": len(new_subtasks),
                    "new_subtasks": [{"id": s.get("id", ""), "title": s.get("title", "")} for s in new_subtasks],
                    "iteration": iteration,
                })
                state["subtasks"].extend(new_subtasks)
                state["research_complete"] = False
            else:
                emit({
                    "type": "reflection-decision", "decision": "research-complete",
                    "iteration": iteration,
                    "total_reports": len(reports),
                    "total_sources": len(state.get("sources", [])),
                })
                state["research_complete"] = True
        except Exception as e:
            logger.warning(f"Reflection failed: {e}")
            emit({"type": "warning", "phase": "reflection", "message": str(e)[:100]})
            state["research_complete"] = True

        persist_state(state, "reflection")
        _emit_progress("reflection")
        _finish_phase("reflection")
        return state

    @traceable(name="synthesize_node")
    def synthesize_node(state: ResearchState) -> ResearchState:
        emit({"type": "phase-start", "phase": "synthesize", "message": f"Synthesizing {len(state.get('subagent_reports', []))} reports..."})
        _emit_progress("synthesize", 0.2)
        state["report"] = synthesize_report(state["user_query"], state["research_plan"], state["subagent_reports"])
        emit({"type": "report-synthesized", "report_length": len(state["report"]), "report_preview": state["report"]})
        _emit_progress("synthesize")
        _finish_phase("synthesize")
        persist_artifact(state.get("run_id", ""), "final_report", state["report"], {})
        persist_state(state, "synthesis")
        return state

    @traceable(name="citation_node")
    def citation_node(state: ResearchState) -> ResearchState:
        source_count = len(state.get("sources", []))
        emit({"type": "phase-start", "phase": "cite", "message": f"Adding citations from {source_count} sources..."})
        _emit_progress("cite", 0.2)
        if source_count == 0:
            emit({"type": "warning", "phase": "cite", "message": "No sources available for citation pass; using synthesized report."})
            state["cited_report"] = state.get("report", "")
        else:
            state["cited_report"] = add_citations(state["report"], state.get("sources", []))
        emit({"type": "citations-added", "cited_report_length": len(state["cited_report"])})
        _emit_progress("cite")
        _finish_phase("cite")
        persist_artifact(state.get("run_id", ""), "cited_report", state["cited_report"], {})
        persist_state(state, "citation")
        return state

    def should_continue(state: ResearchState) -> str:
        return "synthesize" if state.get("research_complete", False) else "subagents"

    # ── wiring ───────────────────────────────────────────────

    graph.add_node("init", init_state)
    graph.add_node("plan", plan_node)
    graph.add_node("split", split_node)
    graph.add_node("scale", scale_node)
    graph.add_node("subagents", subagents_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("cite", citation_node)

    graph.set_entry_point("init")
    graph.add_edge("init", "plan")
    graph.add_edge("plan", "split")
    graph.add_edge("split", "scale")
    graph.add_edge("scale", "subagents")
    graph.add_edge("subagents", "reflection")
    graph.add_conditional_edges("reflection", should_continue, {"synthesize": "synthesize", "subagents": "subagents"})
    graph.add_edge("synthesize", "cite")
    graph.add_edge("cite", END)

    return graph

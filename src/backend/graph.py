"""LangGraph state graph for the research pipeline — all async."""

import asyncio
import json
import logging
import uuid
from typing import Any

from langgraph.graph import END, StateGraph

from .agents import (
    add_citations,
    compute_scaling,
    generate_research_plan,
    run_subagents_parallel,
    split_into_subtasks,
    synthesize_report,
    _chat,
    _extract_json,
)
from .config import get_config
from .models import ResearchState
from .persistence import persist_checkpoint, persist_run, update_run_status
from .prompts import REFLECTION

logger = logging.getLogger(__name__)

PHASE_WEIGHTS = {
    "init": 2, "plan": 8, "split": 5, "scale": 5,
    "subagents": 55, "reflection": 5, "synthesize": 12, "cite": 8,
}
TOTAL_WEIGHT = sum(PHASE_WEIGHTS.values())


async def build_and_run_graph(state: dict[str, Any], on_event) -> dict[str, Any]:
    """Build the LangGraph state machine and run it, emitting events along the way."""

    cfg = get_config()
    completed_weight = 0

    async def _emit(evt: dict[str, Any]) -> None:
        evt.setdefault("run_id", state.get("run_id", ""))
        await on_event(evt)

    async def _init_node(s: ResearchState) -> ResearchState:
        nonlocal completed_weight
        s["run_id"] = s.get("run_id") or str(uuid.uuid4())
        s["events"] = []
        s["errors"] = []
        s["subagent_reports"] = []
        s["sources"] = []
        s["completed_subtasks"] = []
        s["iteration_count"] = 0
        s["max_iterations"] = s.get("max_iterations") or cfg.max_iterations
        s["research_complete"] = False
        s["quality_threshold"] = s.get("quality_threshold") or cfg.quality_threshold
        s["current_quality_score"] = 0.0
        s["memory"] = {}

        await _emit({"type": "phase-update", "phase": "init",
               "message": f"Initialised (provider: {cfg.default_provider}, model: {cfg.default_model})"})
        await _emit({"type": "progress", "phase": "init", "percent": 0})
        completed_weight += PHASE_WEIGHTS["init"]

        await persist_run(s["run_id"], s["user_query"], cfg.default_provider, cfg.default_model)
        await persist_checkpoint(s["run_id"], "init", s)
        return s

    async def _plan_node(s: ResearchState) -> ResearchState:
        nonlocal completed_weight
        await _emit({"type": "phase-update", "phase": "plan", "message": "Generating research plan..."})
        s["research_plan"] = await generate_research_plan(s["user_query"])
        await _emit({"type": "plan-generated", "plan_preview": s["research_plan"][:500],
               "plan_length": len(s["research_plan"])})

        pct = _progress(completed_weight, PHASE_WEIGHTS["plan"])
        await _emit({"type": "progress", "phase": "plan", "percent": pct})
        completed_weight += PHASE_WEIGHTS["plan"]

        await persist_checkpoint(s["run_id"], "plan", s)
        return s

    async def _split_node(s: ResearchState) -> ResearchState:
        nonlocal completed_weight
        await _emit({"type": "phase-update", "phase": "split", "message": "Creating subtasks..."})
        try:
            s["subtasks"] = await split_into_subtasks(s["research_plan"])
        except Exception as e:
            logger.warning("Split failed, fallback: %s", e)
            s["subtasks"] = [{
                "id": "main", "title": s["user_query"][:80],
                "description": s["research_plan"],
                "objective": s["user_query"],
                "output_format": "markdown", "tool_guidance": "web search",
                "source_types": "academic, official, news", "boundaries": "",
            }]

        await _emit({"type": "subtasks-created", "count": len(s["subtasks"]),
               "subtasks": [{"id": t["id"], "title": t["title"],
                             "description": t.get("description", "")[:150],
                             "objective": t.get("objective", "")[:100]}
                            for t in s["subtasks"]]})

        pct = _progress(completed_weight, PHASE_WEIGHTS["split"])
        await _emit({"type": "progress", "phase": "split", "percent": pct})
        completed_weight += PHASE_WEIGHTS["split"]

        await persist_checkpoint(s["run_id"], "split", s)
        return s

    async def _scale_node(s: ResearchState) -> ResearchState:
        nonlocal completed_weight
        await _emit({"type": "phase-update", "phase": "scale", "message": "Estimating complexity..."})
        try:
            s["scaling"] = await compute_scaling(s["user_query"], s["research_plan"])
        except Exception:
            n = len(s.get("subtasks", []))
            s["scaling"] = {"complexity": "moderate", "subagent_count": n,
                           "tool_calls_per_subagent": 10, "target_sources": n * 3}

        subtasks_count = len(s.get("subtasks", []))
        if isinstance(s.get("scaling"), dict):
            s["scaling"]["subagent_count"] = subtasks_count

        await _emit({"type": "scaling-computed", "scaling": s["scaling"]})

        pct = _progress(completed_weight, PHASE_WEIGHTS["scale"])
        await _emit({"type": "progress", "phase": "scale", "percent": pct})
        completed_weight += PHASE_WEIGHTS["scale"]

        await persist_checkpoint(s["run_id"], "scale", s)
        return s

    async def _subagents_node(s: ResearchState) -> ResearchState:
        nonlocal completed_weight
        iteration = s.get("iteration_count", 0)
        completed = set(s.get("completed_subtasks", []))
        to_run = [t for t in s.get("subtasks", []) if t["id"] not in completed]
        if not to_run:
            return s

        await _emit({"type": "phase-update", "phase": "subagents",
               "message": f"Running {len(to_run)} subagents (iteration {iteration + 1})..."})
        await _emit({"type": "subagents-launch", "iteration": iteration + 1, "total_agents": len(to_run),
               "agent_details": [{"id": t["id"], "title": t["title"],
                                  "description": t.get("description", "")[:200]}
                                 for t in to_run]})

        budget = s.get("scaling", {}).get("tool_calls_per_subagent", 15)
        results = await run_subagents_parallel(
            s["user_query"], s["research_plan"], to_run, budget,
        )

        existing_reports = s.get("subagent_reports", [])
        existing_sources = s.get("sources", [])
        s["subagent_reports"] = existing_reports + results["reports"]
        new_completed = [r["subtask_id"] for r in results.get("raw", [])]
        s["completed_subtasks"] = list(completed | set(new_completed))

        all_sources = existing_sources + results["sources"]
        seen: set[str] = set()
        unique: list[dict] = []
        for src in all_sources:
            u = src.get("url")
            if u and u not in seen:
                seen.add(u)
                unique.append(src)
        s["sources"] = unique
        s["iteration_count"] = iteration + 1

        # Emit per-subagent events
        for item in results.get("raw", []):
            await _emit({"type": "subagent-complete",
                   "subtask_id": item["subtask_id"],
                   "subtask_title": item.get("subtask_title", ""),
                   "report_length": len(item.get("report", "")),
                   "sources_count": len(item.get("sources", [])),
                   "evidence_count": item.get("evidence_count", 0)})

        pct = _progress(completed_weight, PHASE_WEIGHTS["subagents"])
        await _emit({"type": "progress", "phase": "subagents", "percent": pct})
        completed_weight += PHASE_WEIGHTS["subagents"]

        await persist_checkpoint(s["run_id"], f"subagents_{iteration}", s)
        return s

    async def _reflection_node(s: ResearchState) -> ResearchState:
        nonlocal completed_weight
        iteration = s.get("iteration_count", 0)
        max_iter = s.get("max_iterations", 2)
        reports = s.get("subagent_reports", [])
        subtasks = s.get("subtasks", [])
        past = ", ".join(f"{t.get('id')}: {t.get('title')}" for t in subtasks)

        if iteration >= max_iter:
            s["research_complete"] = True
            await _emit({"type": "reflection-decision", "decision": "max-iterations-reached", "iteration": iteration})
            return s

        await _emit({"type": "phase-update", "phase": "reflection",
               "message": f"Reflecting (iter {iteration}/{max_iter})..."})

        try:
            truncated = "\n\n".join(r[:3000] for r in reports)
            response = await _chat(
                role="reflection",
                messages=[{
                    "role": "user",
                    "content": REFLECTION.format(
                        user_query=s["user_query"],
                        research_plan=s["research_plan"][:2000],
                        past_subtasks=past,
                        subagent_reports=truncated,
                    ) + "\n\nReturn ONLY valid JSON.",
                }],
            )
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
                await _emit({"type": "reflection-decision", "decision": "gaps-found",
                       "new_subtask_count": len(new_subtasks),
                       "new_subtasks": [{"id": t.get("id", ""), "title": t.get("title", "")}
                                        for t in new_subtasks],
                       "iteration": iteration})
                s["subtasks"].extend(new_subtasks)
                s["research_complete"] = False
            else:
                await _emit({"type": "reflection-decision", "decision": "research-complete",
                       "iteration": iteration,
                       "total_reports": len(reports),
                       "total_sources": len(s.get("sources", []))})
                s["research_complete"] = True
        except Exception as e:
            logger.warning("Reflection failed: %s", e)
            s["research_complete"] = True

        pct = _progress(completed_weight, PHASE_WEIGHTS["reflection"])
        await _emit({"type": "progress", "phase": "reflection", "percent": pct})
        completed_weight += PHASE_WEIGHTS["reflection"]

        await persist_checkpoint(s["run_id"], "reflection", s)
        return s

    async def _synthesize_node(s: ResearchState) -> ResearchState:
        nonlocal completed_weight
        report_count = len(s.get("subagent_reports", []))
        await _emit({"type": "phase-update", "phase": "synthesize",
               "message": f"Synthesizing {report_count} reports..."})

        s["report"] = await synthesize_report(
            s["user_query"], s["research_plan"], s["subagent_reports"],
        )
        await _emit({"type": "report-draft", "content": s["report"][:1000],
               "report_length": len(s["report"])})

        pct = _progress(completed_weight, PHASE_WEIGHTS["synthesize"])
        await _emit({"type": "progress", "phase": "synthesize", "percent": pct})
        completed_weight += PHASE_WEIGHTS["synthesize"]

        await persist_checkpoint(s["run_id"], "synthesis", s)
        return s

    async def _citation_node(s: ResearchState) -> ResearchState:
        nonlocal completed_weight
        source_count = len(s.get("sources", []))
        await _emit({"type": "phase-update", "phase": "cite",
               "message": f"Adding citations from {source_count} sources..."})

        if source_count == 0:
            s["cited_report"] = s.get("report", "")
        else:
            s["cited_report"] = await add_citations(s["report"], s.get("sources", []))
        await _emit({"type": "citations-added", "cited_report_length": len(s["cited_report"])})

        pct = _progress(completed_weight, PHASE_WEIGHTS["cite"])
        await _emit({"type": "progress", "phase": "cite", "percent": pct})
        completed_weight += PHASE_WEIGHTS["cite"]

        await persist_checkpoint(s["run_id"], "citation", s)
        return s

    def _should_continue(s: ResearchState) -> str:
        return "synthesize" if s.get("research_complete", False) else "subagents"

    # Build graph
    graph = StateGraph(ResearchState)
    graph.add_node("init", _init_node)
    graph.add_node("plan", _plan_node)
    graph.add_node("split", _split_node)
    graph.add_node("scale", _scale_node)
    graph.add_node("subagents", _subagents_node)
    graph.add_node("reflection", _reflection_node)
    graph.add_node("synthesize", _synthesize_node)
    graph.add_node("cite", _citation_node)

    graph.set_entry_point("init")
    graph.add_edge("init", "plan")
    graph.add_edge("plan", "split")
    graph.add_edge("split", "scale")
    graph.add_edge("scale", "subagents")
    graph.add_edge("subagents", "reflection")
    graph.add_conditional_edges("reflection", _should_continue, {
        "synthesize": "synthesize", "subagents": "subagents",
    })
    graph.add_edge("synthesize", "cite")
    graph.add_edge("cite", END)

    compiled = graph.compile()
    return await compiled.ainvoke(state)


def _progress(completed: int, current_weight: float, partial: float = 1.0) -> int:
    w = current_weight * partial
    return min(99, int((completed + w) / TOTAL_WEIGHT * 100))

"""LangGraph state graph for the research pipeline — all async."""

import json
import logging
import uuid
from typing import Any

from langgraph.graph import END, StateGraph

from .config import get_config
from .helpers import extract_json
from .llm import chat
from .models import ResearchState
from .persistence import (
    persist_checkpoint,
    persist_run,
    persist_source,
    persist_subagent_report,
    update_run_status,  # noqa: F401 — used by server.py tests
)
from .planning import generate_research_plan, split_into_subtasks
from .prompts import REFLECTION
from .subagent import run_subagents_parallel
from .synthesis import add_citations, synthesize_report

logger = logging.getLogger(__name__)

PHASE_WEIGHTS = {
    "init": 2, "plan": 8, "split": 5,
    "subagents": 60, "reflection": 5, "synthesize": 12, "cite": 8,
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
        plan = await generate_research_plan(s["user_query"])
        s["research_plan"] = plan
        dims = plan.get("dimensions", [])
        preview = json.dumps([d.get("name", "") for d in dims], ensure_ascii=False)
        await _emit({"type": "plan-generated",
               "plan_preview": preview,
               "plan_length": len(json.dumps(plan, ensure_ascii=False)),
               "dimensions": len(dims)})

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
            plan = s.get("research_plan", {})
            plan_text = json.dumps(plan, ensure_ascii=False)
            s["subtasks"] = [{
                "id": "main", "title": s["user_query"][:80],
                "description": plan_text[:500],
                "objective": s["user_query"],
                "output_format": "markdown",
                "dimension": "main",
                "keywords": plan.get("dimensions", [{}])[0].get("keywords", []) if plan.get("dimensions") else [],
                "source_types": "academic, official, news",
                "boundaries": "",
                "estimated_searches": 10,
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

    async def _subagents_node(s: ResearchState) -> ResearchState:
        nonlocal completed_weight
        iteration = s.get("iteration_count", 0)
        completed = set(s.get("completed_subtasks", []))
        to_run = [t for t in s.get("subtasks", []) if t["id"] not in completed]
        if not to_run:
            return s

        # Use subtask-level estimated_searches as budget, default 10
        budget = max(
            t.get("estimated_searches", 10) for t in to_run
        ) if to_run else 10

        await _emit({"type": "phase-update", "phase": "subagents",
               "message": f"Running {len(to_run)} subagents (iteration {iteration + 1})..."})
        await _emit({"type": "subagents-launch", "iteration": iteration + 1, "total_agents": len(to_run),
               "agent_details": [{"id": t["id"], "title": t["title"],
                                  "description": t.get("description", "")[:200]}
                                 for t in to_run]})

        results = await run_subagents_parallel(
            s["user_query"], json.dumps(s.get("research_plan", {}), ensure_ascii=False),
            to_run, budget,
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

        # Emit per-subagent events & persist
        for item in results.get("raw", []):
            await _emit({"type": "subagent-complete",
                   "subtask_id": item["subtask_id"],
                   "subtask_title": item.get("subtask_title", ""),
                   "report_length": len(item.get("report", "")),
                   "sources_count": len(item.get("sources", [])),
                   "evidence_count": item.get("evidence_count", 0)})

            await persist_subagent_report(
                s["run_id"], item["subtask_id"], item.get("report", ""),
                sources_count=len(item.get("sources", [])),
                evidence_count=item.get("evidence_count", 0),
            )

            for src in item.get("sources", []):
                await persist_source(
                    s["run_id"], src.get("url", ""),
                    title=src.get("title", ""),
                    quality_score=src.get("quality_score", 0),
                    domain=src.get("domain", ""),
                    subtask_id=item["subtask_id"],
                )

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
            plan_dims = json.dumps(
                s.get("research_plan", {}).get("dimensions", []),
                ensure_ascii=False
            )
            truncated = "\n\n".join(r[:3000] for r in reports)
            response = await chat(
                role="reflection",
                messages=[{
                    "role": "user",
                    "content": REFLECTION.format(
                        user_query=s["user_query"],
                        research_plan=plan_dims,
                        past_subtasks=past,
                        subagent_reports=truncated,
                    ) + "\n\nReturn ONLY valid JSON.",
                }],
            )
            content = response.strip()
            new_subtasks = []
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                ext = extract_json(content)
                if ext:
                    try:
                        payload = json.loads(ext)
                    except json.JSONDecodeError:
                        payload = {}
                else:
                    payload = {}

            # Extract gap subtasks (limit 3)
            raw_gaps = payload.get("gaps", [])[:3]
            for gap in raw_gaps:
                st = gap.get("subtask", {})
                if st.get("id") and st.get("title"):
                    new_subtasks.append(st)

            overall = payload.get("overall_score", 0)
            s["current_quality_score"] = overall

            if new_subtasks:
                await _emit({"type": "reflection-decision", "decision": "gaps-found",
                       "new_subtask_count": len(new_subtasks),
                       "new_subtasks": [{"id": t.get("id", ""), "title": t.get("title", "")}
                                        for t in new_subtasks],
                       "overall_score": overall,
                       "iteration": iteration})
                s["subtasks"].extend(new_subtasks)
                s["research_complete"] = False
            else:
                await _emit({"type": "reflection-decision", "decision": "research-complete",
                       "iteration": iteration,
                       "total_reports": len(reports),
                       "total_sources": len(s.get("sources", [])),
                       "overall_score": overall})
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

        plan = s.get("research_plan", {})
        plan_text = json.dumps(plan, ensure_ascii=False)
        s["report"] = await synthesize_report(
            s["user_query"], plan_text, s["subagent_reports"],
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

    # Build graph — 7 nodes (scale removed)
    graph = StateGraph(ResearchState)
    graph.add_node("init", _init_node)
    graph.add_node("plan", _plan_node)
    graph.add_node("split", _split_node)
    graph.add_node("subagents", _subagents_node)
    graph.add_node("reflection", _reflection_node)
    graph.add_node("synthesize", _synthesize_node)
    graph.add_node("cite", _citation_node)

    graph.set_entry_point("init")
    graph.add_edge("init", "plan")
    graph.add_edge("plan", "split")
    graph.add_edge("split", "subagents")
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

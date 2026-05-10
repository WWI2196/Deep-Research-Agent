"""Planning-phase pipeline agents: plan, split."""

import json
import logging
from typing import Any

from .helpers import extract_json
from .llm import chat
from .prompts import PLANNER, PLANNER_REACT_SYSTEM, SPLITTER
from .react_agent import run_react_agent
from .tools import Tool, searxng_search_tool, document_hybrid_search_tool

logger = logging.getLogger(__name__)

PLANNER_MAX_STEPS = 6


def _build_planner_tools(
    query_cache: dict[str, dict[str, Any]],
    document_collections: list[str] | None = None,
) -> list[Tool]:
    """Build a minimal tool set for the planner ReAct agent.

    Only searxng_search and (optionally) document_hybrid_search are provided —
    the planner needs to explore the topic landscape, not write reports.
    """

    async def _searxng_search_wrapped(**kwargs: Any) -> dict[str, Any]:
        return await searxng_search_tool(query_cache=query_cache, **kwargs)

    tools = [
        Tool(
            name="searxng_search",
            description=(
                "Search the web to discover the current state of the topic, "
                "key concepts, and recent developments. Returns results with "
                "title, URL, description, and score."
            ),
            params_schema={"query": "str", "limit": "int (optional, default 8)"},
            fn=_searxng_search_wrapped,
        ),
    ]

    if document_collections:
        async def _doc_search_wrapped(**kwargs: Any) -> dict[str, Any]:
            return await document_hybrid_search_tool(
                collection_ids=document_collections, **kwargs,
            )

        tools.append(
            Tool(
                name="document_hybrid_search",
                description=(
                    "Search private document collections for relevant background material. "
                    "Returns document chunks with full text. Highly trusted source."
                ),
                params_schema={"query": "str", "top_k": "int (optional, default 5)"},
                fn=_doc_search_wrapped,
            ),
        )

    return tools


def _parse_plan_json(text: str) -> dict[str, Any] | None:
    """Try to parse a JSON plan from the planner's final_answer."""
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "dimensions" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    extracted = extract_json(text)
    if extracted:
        try:
            parsed = json.loads(extracted)
            if isinstance(parsed, dict) and "dimensions" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    return None


async def generate_research_plan(
    user_query: str,
    document_collections: list[str] | None = None,
) -> dict[str, Any]:
    """Generate a structured research plan using a ReAct agent with search tools.

    The planner performs 2-4 search rounds to explore the topic landscape before
    producing a structured plan. Falls back to single-pass PLANNER prompt on failure.

    Returns a dict with: dimensions, output_structure, methodology.
    """
    query_cache: dict[str, dict[str, Any]] = {}
    tools = _build_planner_tools(query_cache, document_collections)

    user_prompt = (
        f"Research topic: {user_query}\n\n"
        "Explore this topic using the available search tools. "
        "Identify key dimensions, recent developments, and important sub-topics. "
        "Then produce your final structured research plan as JSON."
    )

    try:
        result = await run_react_agent(
            system_prompt=PLANNER_REACT_SYSTEM,
            user_prompt=user_prompt,
            tools=tools,
            chat_fn=chat,
            max_steps=PLANNER_MAX_STEPS,
            temperature=0.3,
            role="planner",
        )

        final_answer = result.get("final_answer", "")
        if final_answer:
            plan = _parse_plan_json(final_answer)
            if plan:
                logger.info(
                    "Planner ReAct completed in %d steps, %d dimensions",
                    result.get("steps_taken", 0),
                    len(plan.get("dimensions", [])),
                )
                return plan

        logger.warning("Planner ReAct did not produce a valid plan, falling back to single-pass")

    except Exception as exc:
        logger.warning("Planner ReAct failed: %s, falling back to single-pass", exc)

    # ── Fallback: single-pass planner ──
    return await _generate_plan_single_pass(user_query, document_collections)


async def _generate_plan_single_pass(
    user_query: str,
    document_collections: list[str] | None = None,
) -> dict[str, Any]:
    """Single-pass planner (original behavior). Used as fallback when ReAct fails."""
    user_content = user_query

    if document_collections:
        try:
            preview_results = await document_hybrid_search_tool(
                query=user_query,
                collection_ids=document_collections,
                top_k=5,
            )
            if preview_results.get("results"):
                preview_lines = ["Available document collections preview:"]
                for i, r in enumerate(preview_results["results"][:5], 1):
                    preview_lines.append(
                        f"{i}. {r.get('title', 'Untitled')} — {r.get('description', '')[:150]}"
                    )
                preview_text = "\n".join(preview_lines)
                user_content = (
                    f"{user_query}\n\n{preview_text}\n\n"
                    "If the preview shows relevant documents, tailor your keywords and source_types "
                    "to leverage these private sources. Mark dimensions that can be primarily "
                    "answered from document library as source_types: document."
                )
        except Exception as exc:
            logger.warning("Planner pre-search failed: %s", exc)

    response = await chat(
        role="planner",
        messages=[
            {"role": "system", "content": PLANNER},
            {"role": "user", "content": user_content + "\n\nReturn valid JSON."},
        ],
    )
    content = response.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        extracted = extract_json(content)
        if extracted:
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                pass

    logger.warning("Planner JSON parse failed, using fallback single-dimension plan")
    return {
        "dimensions": [{
            "name": "main",
            "scope": user_query,
            "source_types": "academic, official, news",
            "keywords": user_query.split()[:5],
        }],
        "output_structure": ["Introduction", "Analysis", "Conclusions"],
        "methodology": "broad web research",
    }


async def split_into_subtasks(research_plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a structured research plan into subtasks.

    On JSON parse failure, feeds the error back to the LLM for one self-heal retry.
    Falls back to a single-subtask plan on double failure.
    """
    plan_text = json.dumps(research_plan, ensure_ascii=False)
    prompt_text = SPLITTER.format(research_plan=plan_text)

    async def _try_split() -> list[dict[str, Any]]:
        response = await chat(
            role="splitter",
            messages=[
                {"role": "system", "content": "You are a research task planner."},
                {"role": "user", "content": prompt_text + "\n\nReturn ONLY valid JSON."},
            ],
        )
        content = response.strip()
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            extracted = extract_json(content)
            if not extracted:
                raise ValueError("Empty or invalid JSON from task splitter.")
            payload = json.loads(extracted)
        return payload["subtasks"]

    def _ensure_ids(subtasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for i, st in enumerate(subtasks):
            if not st.get("id"):
                st["id"] = f"subtask_{i}"
        return subtasks

    try:
        subtasks = await _try_split()
        return _ensure_ids(subtasks)
    except Exception as e:
        logger.warning("Split first attempt failed: %s. Retrying with error feedback...", e)
        retry_prompt = (
            prompt_text +
            f"\n\nYOUR PREVIOUS RESPONSE WAS INVALID. Error: {e}\n"
            "Make sure to return ONLY valid JSON. Do not wrap in markdown code blocks."
        )
        try:
            response = await chat(
                role="splitter",
                messages=[
                    {"role": "system", "content": "You are a research task planner. Return ONLY valid JSON."},
                    {"role": "user", "content": retry_prompt},
                ],
            )
            content = response.strip()
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                extracted = extract_json(content)
                if not extracted:
                    raise ValueError("Empty or invalid JSON from task splitter (retry).")
                payload = json.loads(extracted)
            return _ensure_ids(payload["subtasks"])
        except Exception as e2:
            logger.warning("Split retry also failed: %s. Using fallback.", e2)
            dims = research_plan.get("dimensions", [])
            if dims:
                return [{
                    "id": f"dim_{i}",
                    "title": d.get("name", f"dimension_{i}"),
                    "description": d.get("scope", ""),
                    "objective": d.get("scope", ""),
                    "output_format": "markdown",
                    "dimension": d.get("name", ""),
                    "keywords": d.get("keywords", []),
                    "source_types": d.get("source_types", "academic, official, news"),
                    "boundaries": "",
                    "estimated_searches": 10,
                } for i, d in enumerate(dims)]
            return [{
                "id": "main",
                "title": "Main Research",
                "description": str(research_plan)[:500],
                "objective": str(research_plan)[:300],
                "output_format": "markdown",
                "dimension": "main",
                "keywords": [],
                "source_types": "academic, official, news",
                "boundaries": "",
                "estimated_searches": 10,
            }]

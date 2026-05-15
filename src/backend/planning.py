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

PLANNER_MAX_STEPS = 10


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


def _parse_plan_json(text: str, user_query: str = "") -> dict[str, Any] | None:
    """Try to parse a JSON plan from the planner's final_answer."""
    text = text.strip()
    parsed = None
    
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        extracted = extract_json(text)
        if extracted:
            try:
                parsed = json.loads(extracted)
            except json.JSONDecodeError:
                pass
    
    if not isinstance(parsed, dict) or "dimensions" not in parsed:
        return None
    
    # Ensure requirements field exists with meaningful content
    requirements = parsed.get("requirements", {})
    if not requirements or not any(requirements.values()):
        # Fallback: extract requirements from user_query
        requirements = _extract_requirements_from_query(user_query or parsed.get("user_query", ""))
        parsed["requirements"] = requirements
    
    return parsed


def _extract_requirements_from_query(query: str) -> dict[str, Any]:
    """Extract task requirements from user query using simple heuristics."""
    import re
    
    query_lower = query.lower()
    
    # Detect core objectives
    core_objectives = []
    if any(kw in query_lower for kw in ["比较", "对比", "横向", "compare", "comparison"]):
        core_objectives.append("横向比较多个对象")
    if any(kw in query_lower for kw in ["评估", "评价", "推荐", "evaluate", "recommend", "rank"]):
        core_objectives.append("评估并给出推荐")
    if any(kw in query_lower for kw in ["预测", "forecast", "predict", "趋势", "未来"]):
        core_objectives.append("预测未来趋势")
    if any(kw in query_lower for kw in ["分析", "analyze", "analysis"]):
        core_objectives.append("深入分析")
    if not core_objectives:
        core_objectives.append("Research the given topic thoroughly")
    
    # Detect explicit requirements
    explicit_requirements = []
    
    # Check for specific counts (e.g., "2-3家", "top 10")
    count_patterns = [
        r'(\d+)[-–—](\d+)(?:家|个|款|种|人)?',
        r'(?:top|前)\s*(\d+)',
        r'(\d+)\s*(?:家|个|款|种)',
    ]
    for pattern in count_patterns:
        match = re.search(pattern, query)
        if match:
            explicit_requirements.append(f"需要涉及 {match.group(0)} 的对象")
            break
    
    # Check for dimension comparisons
    dim_keywords = ["维度", "方面", "角度", "指标", "要素"]
    for kw in dim_keywords:
        if kw in query:
            explicit_requirements.append("需要从多个维度进行比较分析")
            break
    
    # Detect scope constraints
    scope_constraints = {"region": "", "time": "", "target": ""}
    
    # Geographic scope
    region_patterns = {
        "中国": r"中国|国内|内地|china",
        "美国": r"美国|usa|america",
        "日本": r"日本|japan",
        "欧洲": r"欧洲|europe|欧盟",
        "全球": r"全球|国际|world|global",
    }
    for region, pattern in region_patterns.items():
        if re.search(pattern, query_lower):
            scope_constraints["region"] = region
            break
    
    # Temporal scope
    time_patterns = {
        "过去五年": r"(?:过去|近|最近)\s*[5五]\s*年",
        "过去十年": r"(?:过去|近|最近)\s*(?:10|十)\s*年",
        "2020-2050": r"2020.*2050|2020.*to.*2050",
        "当前": r"目前|当前|现在|现今",
    }
    for time_range, pattern in time_patterns.items():
        if re.search(pattern, query_lower):
            scope_constraints["time"] = time_range
            break
    
    # Target scope
    if "公司" in query:
        scope_constraints["target"] = "公司"
    elif "产品" in query:
        scope_constraints["target"] = "产品"
    elif "技术" in query:
        scope_constraints["target"] = "技术"
    elif "市场" in query:
        scope_constraints["target"] = "市场"
    
    # Extract sub-questions (split by punctuation)
    sub_questions = []
    # Split by question marks or enumeration markers
    parts = re.split(r'[？?]|(?:\d+[、.])', query)
    for part in parts:
        part = part.strip()
        if len(part) > 10 and len(part) < 200:
            sub_questions.append(part)
    
    if not sub_questions:
        sub_questions = [query]
    
    return {
        "core_objectives": core_objectives,
        "explicit_requirements": explicit_requirements,
        "scope_constraints": scope_constraints,
        "sub_questions": sub_questions[:5],  # Limit to 5 sub-questions
    }


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
            max_search_rounds=6,
        )

        final_answer = result.get("final_answer", "")
        if final_answer:
            plan = _parse_plan_json(final_answer, user_query)
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
        "requirements": {
            "core_objectives": ["Research the given topic thoroughly"],
            "explicit_requirements": [],
            "scope_constraints": {"region": "", "time": "", "target": ""},
            "sub_questions": [user_query],
        },
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
                # Fallback: derive boundaries from other dimensions' scopes
                all_scopes = [d.get("scope", "") for d in dims]
                subtasks = []
                for i, d in enumerate(dims):
                    scope = d.get("scope", "")
                    # Estimate searches based on scope length as a proxy for breadth
                    scope_len = len(scope)
                    if scope_len > 200:
                        est = 10
                    elif scope_len > 100:
                        est = 8
                    else:
                        est = 6
                    # Build boundaries: other dimensions' names/scopes
                    other_names = [
                        other.get("name", f"dim_{j}")
                        for j, other in enumerate(dims) if j != i
                    ]
                    boundaries = (
                        f"Does not cover: {', '.join(other_names)}."
                        if other_names else "Focus only on this dimension."
                    )
                    subtasks.append({
                        "id": f"dim_{i}",
                        "title": d.get("name", f"dimension_{i}"),
                        "description": scope,
                        "objective": scope,
                        "output_format": "markdown",
                        "dimension": d.get("name", ""),
                        "keywords": d.get("keywords", []),
                        "source_types": d.get("source_types", "academic, official, news"),
                        "boundaries": boundaries,
                        "estimated_searches": est,
                    })
                return subtasks
            return [{
                "id": "main",
                "title": "Main Research",
                "description": str(research_plan)[:500],
                "objective": str(research_plan)[:300],
                "output_format": "markdown",
                "dimension": "main",
                "keywords": [],
                "source_types": "academic, official, news",
                "boundaries": "Covers all aspects of the research topic.",
                "estimated_searches": 10,
            }]

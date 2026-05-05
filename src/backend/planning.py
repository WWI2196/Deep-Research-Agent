"""Planning-phase pipeline agents: plan, split."""

import json
import logging
from typing import Any

from .helpers import extract_json
from .llm import chat
from .prompts import PLANNER, SPLITTER

logger = logging.getLogger(__name__)


async def generate_research_plan(user_query: str) -> dict[str, Any]:
    """Generate a structured research plan as JSON.

    Returns a dict with: dimensions, output_structure, methodology.
    Falls back to a single-dimension plan on parse failure.
    """
    response = await chat(
        role="planner",
        messages=[
            {"role": "system", "content": PLANNER},
            {"role": "user", "content": user_query + "\n\nReturn valid JSON."},
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

    try:
        return await _try_split()
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
            return payload["subtasks"]
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

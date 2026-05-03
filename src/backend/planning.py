"""Planning-phase pipeline agents: plan, split, scale."""

import json
from typing import Any

from .helpers import extract_json
from .llm import chat
from .prompts import PLANNER, SCALING, SPLITTER


async def generate_research_plan(user_query: str) -> str:
    return await chat(
        role="planner",
        messages=[
            {"role": "system", "content": PLANNER},
            {"role": "user", "content": user_query},
        ],
    )


async def split_into_subtasks(research_plan: str) -> list[dict[str, Any]]:
    response = await chat(
        role="splitter",
        messages=[
            {"role": "system", "content": SPLITTER},
            {"role": "user", "content": research_plan + "\n\nReturn valid JSON."},
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


async def compute_scaling(user_query: str, research_plan: str) -> dict[str, Any]:
    response = await chat(
        role="scaler",
        messages=[{
            "role": "system",
            "content": SCALING,
        }, {
            "role": "user",
            "content": f"Query: {user_query}\n\nPlan:\n{research_plan}\n\nReturn valid JSON.",
        }],
        temperature=0.1,
    )
    content = response.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        extracted = extract_json(content)
        if not extracted:
            raise ValueError("Empty response from scaler.")
        return json.loads(extracted)

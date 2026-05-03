"""Data models for the research pipeline."""

from typing import Any, TypedDict

from pydantic import BaseModel

# ── Pydantic models for structured LLM output ──

class Subtask(BaseModel):
    id: str
    title: str
    description: str = ""
    objective: str = ""
    output_format: str = "markdown"
    tool_guidance: str = "web search"
    source_types: str = "academic, official, news"
    boundaries: str = ""


class SubtaskList(BaseModel):
    subtasks: list[Subtask]


class ScalingPlan(BaseModel):
    complexity: str
    subagent_count: int
    tool_calls_per_subagent: int
    target_sources: int


# ── Request / response models ──

class ResearchRequest(BaseModel):
    query: str
    max_iterations: int | None = None
    quality_threshold: float | None = None
    provider: str | None = None
    model: str | None = None


class ResearchResponse(BaseModel):
    run_id: str
    status: str


class ConfigUpdateRequest(BaseModel):
    default_provider: str | None = None
    default_model: str | None = None
    max_iterations: int | None = None
    quality_threshold: float | None = None
    roles: dict[str, dict[str, str]] | None = None


# ── LangGraph state ──

class ResearchState(TypedDict, total=False):
    run_id: str
    user_query: str
    research_plan: str
    subtasks: list[dict[str, Any]]
    scaling: dict[str, Any]
    subagent_reports: list[str]
    sources: list[dict[str, Any]]
    report: str
    cited_report: str
    events: list[dict[str, Any]]
    errors: list[str]
    iteration_count: int
    completed_subtasks: list[str]
    max_iterations: int
    research_complete: bool
    memory: dict[str, Any]
    quality_threshold: float
    current_quality_score: float

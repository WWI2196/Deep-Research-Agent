"""Data models for the research pipeline."""

from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


# ---------- Pydantic models for structured LLM output ----------

class Subtask(BaseModel):
    id: str = Field(..., description="Short identifier")
    title: str = Field(..., description="Short descriptive title")
    description: str = Field(..., description="Detailed instructions for the sub-agent")
    objective: str = Field(default="", description="What the subagent should accomplish")
    output_format: str = Field(default="", description="Expected structure of output")
    tool_guidance: str = Field(default="", description="Which tools to prioritize")
    source_types: str = Field(default="", description="Preferred source types")
    boundaries: str = Field(default="", description="What to exclude to avoid overlap")


class SubtaskList(BaseModel):
    subtasks: List[Subtask] = Field(..., description="List of subtasks")


class ScalingPlan(BaseModel):
    complexity: str
    subagent_count: int
    tool_calls_per_subagent: int
    target_sources: int


# ---------- LangGraph state ----------

class ResearchState(TypedDict, total=False):
    run_id: str
    user_query: str
    research_plan: str
    subtasks: List[Dict[str, Any]]
    scaling: Dict[str, Any]
    subagent_reports: List[str]
    sources: List[Dict[str, Any]]
    report: str
    cited_report: str
    events: List[Dict[str, Any]]
    errors: List[str]
    iteration_count: int
    completed_subtasks: List[str]
    max_iterations: int
    research_complete: bool
    memory: Dict[str, Any]
    quality_threshold: float
    current_quality_score: float

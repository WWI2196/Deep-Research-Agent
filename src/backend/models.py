"""Data models for the research pipeline."""

from dataclasses import dataclass
from typing import Any, TypedDict

from pydantic import BaseModel

# ── Depth profile for research depth levels ──

@dataclass
class DepthProfile:
    """Configuration profile for different research depth levels."""

    # Planning
    planner_use_react: bool = True
    planner_max_steps: int = 10
    planner_search_rounds: int = 4
    use_splitter: bool = True
    max_subagents: int = 6

    # Subagent
    search_budget_per_subagent: int = 10
    react_max_steps: int = 15
    evaluate_sources: bool = True
    evaluate_batch_size: int = 20
    fulltext_top_k: int = 5
    max_search_rounds: int = 4
    empty_result_rollback: bool = True

    # Reflection
    max_iterations: int = 2
    quality_threshold: float = 0.65
    max_gaps: int = 3
    min_improvement_gate: float = 0.08

    # Synthesis
    max_input_chars: int = 80000
    continuation_max_rounds: int = 4
    deepen_thin_sections: bool = True
    deepen_max_sections: int = 5
    deepen_char_threshold: int = 800
    deepen_citation_threshold: int = 3
    deepen_concurrency: int = 3
    verify_compliance: bool = True

    # Persistence
    checkpoint_frequency: str = "all"  # "minimal" | "all"
    trace_level: str = "info"  # "warning" | "info" | "debug"


# Depth 1: Quick overview — fast, minimal LLM calls
DEPTH_1_PROFILE = DepthProfile(
    planner_use_react=False,
    planner_max_steps=1,
    planner_search_rounds=0,
    use_splitter=False,
    max_subagents=3,
    search_budget_per_subagent=6,
    react_max_steps=10,
    evaluate_sources=False,
    fulltext_top_k=2,
    max_search_rounds=2,
    empty_result_rollback=False,
    max_iterations=1,
    quality_threshold=0.5,
    max_gaps=0,
    max_input_chars=40000,
    continuation_max_rounds=2,
    deepen_thin_sections=False,
    verify_compliance=False,
    checkpoint_frequency="minimal",
    trace_level="warning",
)

# Depth 2: Balanced — current default behavior
DEPTH_2_PROFILE = DepthProfile()

# Depth 3: Deep research — more thorough, higher quality
DEPTH_3_PROFILE = DepthProfile(
    planner_search_rounds=6,
    planner_max_steps=12,
    max_subagents=8,
    search_budget_per_subagent=15,
    react_max_steps=20,
    fulltext_top_k=8,
    max_search_rounds=6,
    max_iterations=3,
    quality_threshold=0.75,
    max_gaps=5,
    min_improvement_gate=0.05,
    max_input_chars=120000,
    continuation_max_rounds=6,
    deepen_max_sections=8,
    deepen_char_threshold=600,
    deepen_concurrency=5,
    trace_level="debug",
)

DEPTH_PROFILES: dict[int, DepthProfile] = {
    1: DEPTH_1_PROFILE,
    2: DEPTH_2_PROFILE,
    3: DEPTH_3_PROFILE,
}


def get_depth_profile(depth: int) -> DepthProfile:
    """Get the depth profile for a given depth level (1-3)."""
    return DEPTH_PROFILES.get(depth, DEPTH_2_PROFILE)

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
    target_sources: int


# ── Request / response models ──

class ResearchRequest(BaseModel):
    query: str
    depth: int = 2  # Research depth level (1, 2, or 3)
    max_iterations: int | None = None
    quality_threshold: float | None = None
    context_compress_retries: int | None = None
    keep_tool_results: int | None = None
    model: str | None = None
    document_collections: list[str] | None = None
    output_language: str | None = None


class ResearchResponse(BaseModel):
    run_id: str
    status: str


class ConfigUpdateRequest(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    default_model: str | None = None
    quality_threshold: float | None = None
    context_compress_retries: int | None = None
    keep_tool_results: int | None = None
    log_level: str | None = None
    roles: dict[str, dict[str, str]] | None = None


class CollectionCreateRequest(BaseModel):
    name: str
    description: str = ""


class CollectionUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None


# ── LangGraph state ──

class ResearchState(TypedDict, total=False):
    run_id: str
    user_query: str
    depth: int  # Research depth level (1, 2, or 3)
    depth_profile: DepthProfile  # Resolved depth profile
    research_plan: dict[str, Any]  # structured plan from planner
    subtasks: list[dict[str, Any]]
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
    synthesis_retry_count: int
    synthesis_failure_summary: str
    context_compress_retries: int
    keep_tool_results: int
    query_cache: dict[str, list[dict[str, Any]]]
    document_collections: list[str]
    output_language: str
    bench_format: bool
    # Task requirements extracted from user query
    requirements: dict[str, Any]
    # Agentic RAG negotiation fields
    gap_instructions: list[dict[str, Any]]
    tool_call_history: list[dict[str, Any]]
    # Internal mapping for precise deduplication during reflection re-runs
    _subtask_report_map: dict[str, str]

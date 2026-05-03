"""Backward-compatible re-export shim.

Prefer importing directly from llm, helpers, planning, subagent, synthesis.
"""

# ruff: noqa: F401

from .helpers import (
    clean_think_tags as _clean_think_tags,
    enforce_source_diversity as _enforce_source_diversity,
    extract_json as _extract_json,
    has_clean_ending as _has_clean_ending,
    needs_continuation as _needs_continuation,
    normalize_search_item as _normalize_search_item,
    pick_first_nonempty as _pick_first_nonempty,
)
from .llm import chat as _chat, invalidate_provider_cache
from .planning import compute_scaling, generate_research_plan, split_into_subtasks
from .subagent import (
    _refine_queries_if_needed,
    batch_evaluate_sources,
    generate_search_queries,
    run_subagent,
    run_subagents_parallel,
)
from .synthesis import (
    _continue_if_truncated,
    add_citations,
    synthesize_report,
)

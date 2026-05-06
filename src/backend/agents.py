"""Backward-compatible re-export shim.

Prefer importing directly from llm, helpers, planning, subagent, synthesis.
"""

# ruff: noqa: F401

from .helpers import (
    clean_think_tags as _clean_think_tags,
)
from .helpers import (
    enforce_source_diversity as _enforce_source_diversity,
)
from .helpers import (
    extract_json as _extract_json,
)
from .helpers import (
    has_clean_ending as _has_clean_ending,
)
from .helpers import (
    needs_continuation as _needs_continuation,
)
from .helpers import (
    normalize_search_item as _normalize_search_item,
)
from .helpers import (
    pick_first_nonempty as _pick_first_nonempty,
)
from .llm import chat as _chat
from .llm import invalidate_provider_cache
from .planning import generate_research_plan, split_into_subtasks
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

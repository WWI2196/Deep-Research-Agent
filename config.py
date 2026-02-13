import os
from typing import List, Optional


class ModelConfig:
    """
    Model configuration with multi-provider fallback support.

    The system will try each (model_id, provider) pair in order until one works.
    This handles 402 (credits depleted), 404 (model not found), and other errors
    by gracefully falling to the next option.
    """

    def __init__(
        self,
        model_id: str,
        provider: str = "novita",
        fallbacks: Optional[List[dict]] = None,
    ):
        self.model_id = model_id
        self.provider = provider
        # List of {"model_id": ..., "provider": ...} dicts to try if primary fails
        self.fallbacks = fallbacks or []

    def get_all_options(self) -> List[dict]:
        """Return primary + fallback options in order."""
        options = [{"model_id": self.model_id, "provider": self.provider}]
        options.extend(self.fallbacks)
        return options


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Each config has a primary model + fallback chain.
# The system tries each in order on 402/404/5xx errors.
#
# To use different providers, set env vars:
#   HF_TOKEN          – HuggingFace API token
#   OPENROUTER_API_KEY – OpenRouter API key (optional fallback)
# =============================================================================

# 1. PLANNER: Generates the initial high-level research plan
PLANNER_CONFIG = ModelConfig(
    model_id="Qwen/Qwen3-235B-A22B-Thinking-2507",
    provider="novita",
    fallbacks=[
        {"model_id": "deepseek-ai/DeepSeek-R1-0528", "provider": "novita"},
        {"model_id": "Qwen/Qwen2.5-72B-Instruct", "provider": "novita"},
        {"model_id": "meta-llama/Llama-3.3-70B-Instruct", "provider": "sambanova"},
        {"model_id": "Qwen/Qwen2.5-72B-Instruct", "provider": "auto"},
    ],
)

# 2. TASK SPLITTER: Breaks the plan into specific JSON subtasks
TASK_SPLITTER_CONFIG = ModelConfig(
    model_id="zai-org/GLM-4.7",
    provider="novita",
    fallbacks=[
        {"model_id": "Qwen/Qwen3-235B-A22B-Instruct-2507", "provider": "novita"},
        {"model_id": "Qwen/Qwen2.5-72B-Instruct", "provider": "novita"},
        {"model_id": "meta-llama/Llama-3.3-70B-Instruct", "provider": "sambanova"},
    ],
)

# 3. SCALING: Estimates the complexity and resource budget
SCALING_CONFIG = ModelConfig(
    model_id="Qwen/Qwen3-235B-A22B-Thinking-2507",
    provider="novita",
    fallbacks=[
        {"model_id": "Qwen/Qwen2.5-72B-Instruct", "provider": "novita"},
        {"model_id": "meta-llama/Llama-3.3-70B-Instruct", "provider": "sambanova"},
    ],
)

# 4. SUBAGENTS: Search, extract, and write partial reports
SUBAGENT_CONFIG = ModelConfig(
    model_id="MiniMaxAI/MiniMax-M1-80k",
    provider="novita",
    fallbacks=[
        {"model_id": "Qwen/Qwen3-235B-A22B-Instruct-2507", "provider": "novita"},
        {"model_id": "Qwen/Qwen2.5-72B-Instruct", "provider": "novita"},
        {"model_id": "meta-llama/Llama-3.3-70B-Instruct", "provider": "sambanova"},
    ],
)

# 5. COORDINATOR: Synthesizes final report
COORDINATOR_CONFIG = ModelConfig(
    model_id="MiniMaxAI/MiniMax-M1-80k",
    provider="novita",
    fallbacks=[
        {"model_id": "Qwen/Qwen3-235B-A22B-Instruct-2507", "provider": "novita"},
        {"model_id": "Qwen/Qwen2.5-72B-Instruct", "provider": "novita"},
    ],
)

# 6. REFLECTION: Reviews progress and generates new subtasks
REFLECTION_CONFIG = ModelConfig(
    model_id="Qwen/Qwen3-235B-A22B-Instruct-2507",
    provider="novita",
    fallbacks=[
        {"model_id": "Qwen/Qwen2.5-72B-Instruct", "provider": "novita"},
        {"model_id": "meta-llama/Llama-3.3-70B-Instruct", "provider": "sambanova"},
    ],
)

# 7. CITATION: Adds inline citations
CITATION_CONFIG = ModelConfig(
    model_id="Qwen/Qwen3-235B-A22B-Instruct-2507",
    provider="novita",
    fallbacks=[
        {"model_id": "Qwen/Qwen2.5-72B-Instruct", "provider": "novita"},
    ],
)

# 8. SOURCE EVALUATOR: Scores and filters search results by quality
SOURCE_EVALUATOR_CONFIG = ModelConfig(
    model_id="Qwen/Qwen3-235B-A22B-Instruct-2507",
    provider="novita",
    fallbacks=[
        {"model_id": "Qwen/Qwen2.5-72B-Instruct", "provider": "novita"},
    ],
)

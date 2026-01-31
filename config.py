import os

class ModelConfig:
    def __init__(self, model_id: str, provider: str = "auto"):
        self.model_id = model_id
        self.provider = provider

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# 1. PLANNER: Generates the initial high-level research plan
# Recommended: Reasoning models with strong "thinking" capabilities
PLANNER_CONFIG = ModelConfig(
    model_id="Qwen/Qwen3-235B-A22B-Thinking-2507",
    provider="auto"
)

# 2. TASK SPLITTER: Breaks the plan into specific JSON subtasks
# Recommended: Models with strong instruction following and JSON support 
TASK_SPLITTER_CONFIG = ModelConfig(
    model_id="zai-org/GLM-4.7",
    provider="novita"
)

# 3. SCALING: Estimates the complexity and resource budget
# Recommended: Reasoning models
SCALING_CONFIG = ModelConfig(
    model_id="Qwen/Qwen3-235B-A22B-Thinking-2507",
    provider="novita"
)

# 4. SUBAGENTS: Search, extract, and write partial reports
# Recommended: Long context models
SUBAGENT_CONFIG = ModelConfig(
    model_id="MiniMaxAI/MiniMax-M1-80k",
    provider="novita"
)

# 5. COORDINATOR: Synthesizes final report
# Recommended: Large context models
COORDINATOR_CONFIG = ModelConfig(
    model_id="MiniMaxAI/MiniMax-M1-80k",
    provider="novita"
)

# 6. CITATION: Adds inline citations
# Recommended: Strong instruction following
CITATION_CONFIG = ModelConfig(
    model_id="Qwen/Qwen3-235B-A22B-Instruct-2507",
    provider="novita"
)

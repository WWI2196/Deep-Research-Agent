"""Centralised configuration.

Provider and model selection is driven by environment variables.
Supports a single default provider for all roles, with optional
per-role overrides.

Env vars
--------
LLM_PROVIDER           Default provider (gemini / openai / anthropic / huggingface)
LLM_MODEL              Default model   (auto-selected per provider if empty)

GEMINI_API_KEY         Google Gemini
OPENAI_API_KEY         OpenAI
ANTHROPIC_API_KEY      Anthropic
HF_TOKEN               HuggingFace

PLANNER_PROVIDER       Per-role override example
PLANNER_MODEL          Per-role override example
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

from dotenv import load_dotenv

load_dotenv()

# ---- sensible defaults per provider ----
DEFAULT_MODELS: Dict[str, str] = {
    "gemini": "gemini-3-pro-preview",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "huggingface": "Qwen/Qwen2.5-72B-Instruct",
}
# All available models per provider (for UI/config)
AVAILABLE_MODELS: Dict[str, list] = {
    "gemini": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-pro-preview"],
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1", "o1-mini"],
    "anthropic": ["claude-sonnet-4-20250514", "claude-haiku-3-5-20241022", "claude-opus-4-20250514"],
    "huggingface": ["Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"],
}
ROLES = [
    "planner", "splitter", "scaler", "subagent",
    "evaluator", "coordinator", "citation", "reflection",
]


@dataclass
class RoleConfig:
    provider: str
    model: str
    temperature: float = 0.2


@dataclass
class AppConfig:
    default_provider: str = "gemini"
    default_model: str = ""
    max_iterations: int = 2
    quality_threshold: float = 0.7
    roles: Dict[str, RoleConfig] = field(default_factory=dict)

    def get_role(self, name: str) -> RoleConfig:
        """Return config for *name*, falling back to the global default."""
        if name in self.roles:
            return self.roles[name]
        return RoleConfig(
            provider=self.default_provider,
            model=self.default_model or DEFAULT_MODELS.get(self.default_provider, ""),
        )


def load_config() -> AppConfig:
    provider = os.getenv("LLM_PROVIDER", "gemini")
    model = os.getenv("LLM_MODEL", DEFAULT_MODELS.get(provider, ""))

    cfg = AppConfig(
        default_provider=provider,
        default_model=model,
        max_iterations=int(os.getenv("MAX_ITERATIONS", "3")),
        quality_threshold=float(os.getenv("QUALITY_THRESHOLD", "0.7")),
    )

    for role in ROLES:
        pfx = role.upper()
        p = os.getenv(f"{pfx}_PROVIDER")
        m = os.getenv(f"{pfx}_MODEL")
        t = os.getenv(f"{pfx}_TEMPERATURE")
        if p or m or t:
            cfg.roles[role] = RoleConfig(
                provider=p or provider,
                model=m or model,
                temperature=float(t) if t else 0.2,
            )
    return cfg


_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config():
    global _config
    _config = load_config()

"""Configuration system.

Priority: environment variable > config.yaml > built-in default.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

CONFIG_DIR = Path.home() / ".deep-research"
CONFIG_PATH = CONFIG_DIR / "config.yaml"

ROLES = ["planner", "splitter", "scaler", "subagent", "evaluator", "coordinator", "reflection", "citation"]


@dataclass
class RoleConfig:
    model: str
    temperature: float = 0.2


@dataclass
class AppConfig:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    default_model: str = "gpt-4o"
    quality_threshold: float = 0.65
    max_sources_per_domain: int = 5
    context_compress_retries: int = 1
    keep_tool_results: int = 5
    max_evidence_tokens: int = 8000
    max_evidence_per_item: int = 3000
    source_type_quotas: dict[str, int] = field(default_factory=lambda: {"document": 3, "web": 8})
    min_source_per_type: dict[str, int] = field(default_factory=lambda: {"document": 1, "web": 2})
    log_level: str = "info"
    output_language: str = "zh"
    roles: dict[str, RoleConfig] = field(default_factory=dict)
    agentic_rag_enabled: bool = True

    def get_role(self, name: str) -> RoleConfig:
        if name in self.roles:
            return self.roles[name]
        return RoleConfig(model=self.default_model)

    def to_dict(self) -> dict:
        return {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "default_model": self.default_model,
            "quality_threshold": self.quality_threshold,
            "context_compress_retries": self.context_compress_retries,
            "keep_tool_results": self.keep_tool_results,
            "max_evidence_tokens": self.max_evidence_tokens,
            "max_evidence_per_item": self.max_evidence_per_item,
            "source_type_quotas": self.source_type_quotas,
            "min_source_per_type": self.min_source_per_type,
            "log_level": self.log_level,
            "output_language": self.output_language,
            "agentic_rag_enabled": self.agentic_rag_enabled,
            "roles": {
                name: {"model": rc.model}
                for name, rc in self.roles.items()
            },
        }


def _resolve_env_vars(value: str) -> str:
    pattern = re.compile(r"\$\{(\w+)}")

    def _replace(match):
        return os.environ.get(match.group(1), "")

    return pattern.sub(_replace, value)


def _load_yaml_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f) or {}

    def _resolve(obj):
        if isinstance(obj, str):
            return _resolve_env_vars(obj)
        if isinstance(obj, dict):
            return {k: _resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve(v) for v in obj]
        return obj

    return _resolve(raw)


def load_config() -> AppConfig:
    yaml_cfg = _load_yaml_config()
    llm_cfg = yaml_cfg.get("llm", {}) or {}

    base_url = (
        os.getenv("LLM_BASE_URL")
        or llm_cfg.get("base_url", "")
        or "https://api.openai.com/v1"
    )
    api_key = (
        os.getenv("LLM_API_KEY")
        or llm_cfg.get("api_key", "")
    )
    default_model = (
        os.getenv("LLM_MODEL")
        or llm_cfg.get("model", "")
        or "gpt-4o"
    )

    if isinstance(base_url, str):
        base_url = _resolve_env_vars(base_url)
    if isinstance(api_key, str):
        api_key = _resolve_env_vars(api_key)

    research_yaml = yaml_cfg.get("research", {}) or {}

    def _research_float(env_key: str, yaml_key: str, default: float) -> float:
        env_val = os.getenv(env_key)
        if env_val is not None:
            return float(env_val)
        yaml_val = research_yaml.get(yaml_key)
        if yaml_val is not None:
            return float(yaml_val)
        return default

    def _research_int(env_key: str, yaml_key: str, default: int) -> int:
        env_val = os.getenv(env_key)
        if env_val is not None:
            return int(env_val)
        yaml_val = research_yaml.get(yaml_key)
        if yaml_val is not None:
            return int(yaml_val)
        return default

    def _research_dict(env_key: str, yaml_key: str, default: dict) -> dict:
        env_val = os.getenv(env_key)
        if env_val is not None:
            try:
                import json as _json
                return _json.loads(env_val)
            except Exception:
                pass
        yaml_val = research_yaml.get(yaml_key)
        if yaml_val is not None and isinstance(yaml_val, dict):
            return yaml_val
        return default

    cfg = AppConfig(
        base_url=base_url,
        api_key=api_key,
        default_model=default_model,
        quality_threshold=_research_float("QUALITY_THRESHOLD", "quality_threshold", 0.6),
        max_sources_per_domain=_research_int("MAX_SOURCES_PER_DOMAIN", "max_sources_per_domain", 5),
        context_compress_retries=_research_int("CONTEXT_COMPRESS_RETRIES", "context_compress_retries", 1),
        keep_tool_results=_research_int("KEEP_TOOL_RESULTS", "keep_tool_results", 5),
        max_evidence_tokens=_research_int("MAX_EVIDENCE_TOKENS", "max_evidence_tokens", 8000),
        max_evidence_per_item=_research_int("MAX_EVIDENCE_PER_ITEM", "max_evidence_per_item", 3000),
        source_type_quotas=_research_dict("SOURCE_TYPE_QUOTAS", "source_type_quotas", {"document": 3, "web": 8}),
        min_source_per_type=_research_dict("MIN_SOURCE_PER_TYPE", "min_source_per_type", {"document": 1, "web": 2}),
        log_level=(
            os.getenv("LOG_LEVEL")
            or research_yaml.get("log_level", "")
            or "info"
        ).lower(),
        output_language=(
            os.getenv("OUTPUT_LANGUAGE")
            or research_yaml.get("output_language", "")
            or "zh"
        ).lower(),
        agentic_rag_enabled=(
            os.getenv("AGENTIC_RAG_ENABLED", "").lower() in ("1", "true", "yes")
            or research_yaml.get("agentic_rag_enabled", False) is True
        ),
    )

    yaml_roles = yaml_cfg.get("roles", {}) or {}
    for role in ROLES:
        env_pfx = role.upper()
        model = os.getenv(f"{env_pfx}_MODEL") or yaml_roles.get(role, {}).get("model")
        if model:
            cfg.roles[role] = RoleConfig(model=model)

    return cfg


def save_config(cfg: AppConfig) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    existing = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            existing = yaml.safe_load(f) or {}

    existing["llm"] = {
        "base_url": cfg.base_url,
        "api_key": cfg.api_key,
        "model": cfg.default_model,
    }
    existing["roles"] = {
        name: {"model": rc.model}
        for name, rc in cfg.roles.items()
    }
    existing["research"] = {
        "quality_threshold": cfg.quality_threshold,
        "context_compress_retries": cfg.context_compress_retries,
        "keep_tool_results": cfg.keep_tool_results,
        "max_evidence_tokens": cfg.max_evidence_tokens,
        "max_evidence_per_item": cfg.max_evidence_per_item,
        "source_type_quotas": cfg.source_type_quotas,
        "min_source_per_type": cfg.min_source_per_type,
        "log_level": cfg.log_level,
        "output_language": cfg.output_language,
        "agentic_rag_enabled": cfg.agentic_rag_enabled,
    }

    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(existing, f, default_flow_style=False, allow_unicode=True)


_config: AppConfig | None = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config() -> AppConfig:
    global _config
    _config = load_config()
    return _config

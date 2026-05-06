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

# ── built-in providerefaults ─────────────────────────────────

BUILTIN_PROVIDERS = {
    "mimo": {
        "type": "openai",
        "base_url": "https://api.xiaomimimo.com/v1",
        "api_key_env": "MIMO_API_KEY",
    },
    "openai": {
        "type": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "anthropic": {
        "type": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "gemini": {
        "type": "openai",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
    },
    "deepseek": {
        "type": "openai",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "openrouter": {
        "type": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
}

ROLES = ["planner", "splitter", "scaler", "subagent", "evaluator", "coordinator", "reflection", "citation"]


@dataclass
class ProviderConfig:
    name: str
    type: str       # "openai" | "anthropic"
    base_url: str = ""
    api_key: str = ""


@dataclass
class RoleConfig:
    provider: str
    model: str
    temperature: float = 0.2


@dataclass
class AppConfig:
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    default_provider: str = "openai"
    default_model: str = "gpt-4o"
    max_iterations: int = 3
    quality_threshold: float = 0.7
    max_sources_per_domain: int = 3
    tool_calls_per_subagent: int = 15
    context_compress_retries: int = 1
    keep_tool_results: int = 5
    roles: dict[str, RoleConfig] = field(default_factory=dict)

    def get_role(self, name: str) -> RoleConfig:
        if name in self.roles:
            return self.roles[name]
        return RoleConfig(provider=self.default_provider, model=self.default_model)

    def get_provider(self, name: str) -> ProviderConfig:
        return self.providers.get(name, self.providers.get(self.default_provider))

    def to_dict(self) -> dict:
        return {
            "default_provider": self.default_provider,
            "default_model": self.default_model,
            "max_iterations": self.max_iterations,
            "quality_threshold": self.quality_threshold,
            "context_compress_retries": self.context_compress_retries,
            "keep_tool_results": self.keep_tool_results,
            "roles": {
                name: {"provider": rc.provider, "model": rc.model}
                for name, rc in self.roles.items()
            },
        }


def _resolve_env_vars(value: str) -> str:
    pattern = re.compile(r"\$\{(\w+)\}")

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
    yaml_providers = yaml_cfg.get("providers", {}) or {}

    # ── Build provider list ──
    providers: dict[str, ProviderConfig] = {}

    # Start with built-in providers
    for name, pdef in BUILTIN_PROVIDERS.items():
        yp = yaml_providers.pop(name, {})  # pop so remaining are user-defined only

        # api_key: env(api_key_env) > yaml(api_key) > env(builtin api_key_env) > ""
        api_key_env = yp.get("api_key_env", pdef.get("api_key_env", ""))
        api_key = (
            os.environ.get(api_key_env, "")
            or yp.get("api_key", "")
            or os.environ.get(pdef.get("api_key_env", ""), "")
        )

        providers[name] = ProviderConfig(
            name=name,
            type=yp.get("type", pdef["type"]),
            base_url=yp.get("base_url", pdef.get("base_url", "")),
            api_key=api_key,
        )

    # Remaining yaml providers are fully user-defined
    for name, pdef in yaml_providers.items():
        api_key_env = pdef.get("api_key_env", "")
        api_key = os.environ.get(api_key_env, "") or pdef.get("api_key", "")
        if isinstance(api_key, str) and api_key.startswith("${"):
            api_key = _resolve_env_vars(api_key)
        providers[name] = ProviderConfig(
            name=name,
            type=pdef.get("type", "openai"),
            base_url=pdef.get("base_url", ""),
            api_key=api_key,
        )

    # ── Default provider / model: env > yaml > built-in ──
    default_provider = (
        os.getenv("LLM_PROVIDER")
        or yaml_cfg.get("default", {}).get("provider", "")
    )
    if not default_provider:
        for name, pc in providers.items():
            if pc.api_key:
                default_provider = name
                break
    if not default_provider:
        default_provider = "mimo"

    default_model = (
        os.getenv("LLM_MODEL")
        or yaml_cfg.get("default", {}).get("model", "mimo-v2.5-pro")
    )

    # ── Research params: env > yaml > built-in ──
    research_yaml = yaml_cfg.get("research", {}) or {}

    def _research_int(env_key: str, yaml_key: str, default: int) -> int:
        env_val = os.getenv(env_key)
        if env_val is not None:
            return int(env_val)
        yaml_val = research_yaml.get(yaml_key)
        if yaml_val is not None:
            return int(yaml_val)
        return default

    def _research_float(env_key: str, yaml_key: str, default: float) -> float:
        env_val = os.getenv(env_key)
        if env_val is not None:
            return float(env_val)
        yaml_val = research_yaml.get(yaml_key)
        if yaml_val is not None:
            return float(yaml_val)
        return default

    cfg = AppConfig(
        providers=providers,
        default_provider=default_provider,
        default_model=default_model,
        max_iterations=_research_int("MAX_ITERATIONS", "max_iterations", 3),
        quality_threshold=_research_float("QUALITY_THRESHOLD", "quality_threshold", 0.7),
        max_sources_per_domain=_research_int("MAX_SOURCES_PER_DOMAIN", "max_sources_per_domain", 3),
        tool_calls_per_subagent=_research_int("TOOL_CALLS_PER_SUBAGENT", "tool_calls_per_subagent", 15),
        context_compress_retries=_research_int("CONTEXT_COMPRESS_RETRIES", "context_compress_retries", 1),
        keep_tool_results=_research_int("KEEP_TOOL_RESULTS", "keep_tool_results", 5),
    )

    # ── Role overrides: env > yaml > (follow default) ──
    yaml_roles = yaml_cfg.get("roles", {}) or {}
    for role in ROLES:
        env_pfx = role.upper()
        provider = os.getenv(f"{env_pfx}_PROVIDER") or yaml_roles.get(role, {}).get("provider")
        model = os.getenv(f"{env_pfx}_MODEL") or yaml_roles.get(role, {}).get("model")
        if provider or model:
            cfg.roles[role] = RoleConfig(
                provider=provider or default_provider,
                model=model or default_model,
            )

    return cfg


def save_config(cfg: AppConfig) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # Preserve existing sections (providers, search, etc.) that aren't managed here
    existing = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            existing = yaml.safe_load(f) or {}

    # Merge managed sections on top, keeping unmanaged keys
    existing["default"] = {"provider": cfg.default_provider, "model": cfg.default_model}
    existing["roles"] = {
        name: {"provider": rc.provider, "model": rc.model}
        for name, rc in cfg.roles.items()
    }
    existing["research"] = {
        "max_iterations": cfg.max_iterations,
        "quality_threshold": cfg.quality_threshold,
        "context_compress_retries": cfg.context_compress_retries,
        "keep_tool_results": cfg.keep_tool_results,
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

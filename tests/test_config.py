"""Tests for configuration system."""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch


@pytest.fixture(autouse=True)
def clear_config_cache():
    from src.backend import config
    config._config = None
    yield
    config._config = None


def test_load_config_builtin_providers():
    with patch("src.backend.config._load_yaml_config", return_value={}), \
         patch.dict(os.environ, {}, clear=True):
        from src.backend.config import load_config
        cfg = load_config()
        assert "openai" in cfg.providers
        assert "anthropic" in cfg.providers
        assert "deepseek" in cfg.providers
        assert cfg.providers["openai"].type == "openai"
        assert cfg.providers["anthropic"].type == "anthropic"
        assert cfg.providers["deepseek"].base_url == "https://api.deepseek.com/v1"


def test_default_provider_picks_first_with_key():
    with patch("src.backend.config._load_yaml_config", return_value={}), \
         patch.dict(os.environ, {"MIMO_API_KEY": "sk-test"}, clear=True):
        from src.backend.config import load_config
        cfg = load_config()
        assert cfg.default_provider == "mimo"


def test_role_override_from_env():
    with patch("src.backend.config._load_yaml_config", return_value={}), \
         patch.dict(os.environ, {
             "OPENAI_API_KEY": "sk-test",
             "PLANNER_PROVIDER": "openrouter",
             "PLANNER_MODEL": "openai/gpt-4o",
         }, clear=True):
        from src.backend.config import load_config
        cfg = load_config()
        assert "planner" in cfg.roles
        assert cfg.roles["planner"].provider == "openrouter"
        assert cfg.roles["planner"].model == "openai/gpt-4o"


def test_get_role_falls_back_to_default():
    with patch("src.backend.config._load_yaml_config", return_value={}), \
         patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
        from src.backend.config import load_config
        cfg = load_config()
        role = cfg.get_role("subagent")
        assert role.provider == cfg.default_provider
        assert role.model == cfg.default_model


# ── New tests: env var resolution ──────────────────────────────

def test_resolve_env_vars_basic():
    from src.backend.config import _resolve_env_vars
    with patch.dict(os.environ, {"TEST_VAR": "test_value"}, clear=True):
        result = _resolve_env_vars("prefix-${TEST_VAR}-suffix")
        assert result == "prefix-test_value-suffix"


def test_resolve_env_vars_unset_var():
    from src.backend.config import _resolve_env_vars
    with patch.dict(os.environ, {}, clear=True):
        result = _resolve_env_vars("prefix-${UNSET_VAR}-suffix")
        assert result == "prefix--suffix"


def test_resolve_env_vars_multiple():
    from src.backend.config import _resolve_env_vars
    with patch.dict(os.environ, {"A": "1", "B": "2"}, clear=True):
        result = _resolve_env_vars("${A}_${B}")
        assert result == "1_2"


# ── API key priority: env(api_key_env) > yaml(api_key) > env(builtin_api_key_env) ──

def test_api_key_priority_env_wins():
    with patch("src.backend.config._load_yaml_config", return_value={
        "providers": {"openai": {"api_key": "yaml-key"}}
    }), patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=True):
        from src.backend.config import load_config
        cfg = load_config()
        assert cfg.providers["openai"].api_key == "env-key"


def test_api_key_priority_yaml_wins_over_builtin_env():
    with patch("src.backend.config._load_yaml_config", return_value={
        "providers": {"openai": {"api_key": "yaml-key"}}
    }), patch.dict(os.environ, {}, clear=True):
        from src.backend.config import load_config
        cfg = load_config()
        assert cfg.providers["openai"].api_key == "yaml-key"


def test_api_key_falls_back_to_builtin_env():
    with patch("src.backend.config._load_yaml_config", return_value={}), \
         patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=True):
        from src.backend.config import load_config
        cfg = load_config()
        assert cfg.providers["openai"].api_key == "env-key"


# ── User-defined providers ─────────────────────────────────────

def test_user_defined_provider_in_yaml():
    yaml_cfg = {
        "providers": {
            "custom-llm": {
                "type": "openai",
                "base_url": "https://custom.api/v1",
                "api_key": "custom-key",
            }
        }
    }
    with patch("src.backend.config._load_yaml_config", return_value=yaml_cfg), \
         patch.dict(os.environ, {}, clear=True):
        from src.backend.config import load_config
        cfg = load_config()
        assert "custom-llm" in cfg.providers
        assert cfg.providers["custom-llm"].type == "openai"
        assert cfg.providers["custom-llm"].base_url == "https://custom.api/v1"
        assert cfg.providers["custom-llm"].api_key == "custom-key"


# ── Research params: env > yaml > builtin ──────────────────────

def test_research_params_env_wins():
    with patch("src.backend.config._load_yaml_config", return_value={
        "research": {"max_iterations": 4}
    }), patch.dict(os.environ, {"MAX_ITERATIONS": "7"}, clear=True):
        from src.backend.config import load_config
        cfg = load_config()
        assert cfg.max_iterations == 7


def test_research_params_yaml_wins_over_default():
    with patch("src.backend.config._load_yaml_config", return_value={
        "research": {"quality_threshold": 0.85}
    }), patch.dict(os.environ, {}, clear=True):
        from src.backend.config import load_config
        cfg = load_config()
        assert cfg.quality_threshold == 0.85


def test_research_params_defaults():
    with patch("src.backend.config._load_yaml_config", return_value={}), \
         patch.dict(os.environ, {}, clear=True):
        from src.backend.config import load_config
        cfg = load_config()
        assert cfg.max_iterations == 3
        assert cfg.quality_threshold == 0.6
        assert cfg.max_sources_per_domain == 3
        assert cfg.tool_calls_per_subagent == 15


# ── LLM_PROVIDER env overrides default provider selection ──────

def test_llm_provider_env_overrides_detection():
    with patch("src.backend.config._load_yaml_config", return_value={}), \
         patch.dict(os.environ, {
             "LLM_PROVIDER": "deepseek",
             "DEEPSEEK_API_KEY": "sk-ds",
         }, clear=True):
        from src.backend.config import load_config
        cfg = load_config()
        assert cfg.default_provider == "deepseek"


# ── LLM_MODEL env overrides default model ──────────────────────

def test_llm_model_env_overrides():
    with patch("src.backend.config._load_yaml_config", return_value={}), \
         patch.dict(os.environ, {
             "LLM_MODEL": "gpt-4o-mini",
             "OPENAI_API_KEY": "sk-test",
         }, clear=True):
        from src.backend.config import load_config
        cfg = load_config()
        assert cfg.default_model == "gpt-4o-mini"


# ── save_config + reload_config round-trip ─────────────────────

def test_save_and_reload_roundtrip():
    from src.backend import config as cfg_mod
    import yaml

    # Build a known config
    cfg = cfg_mod.AppConfig(
        providers={},
        default_provider="openai",
        default_model="gpt-4o",
        max_iterations=5,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = cfg_mod.CONFIG_DIR
        original_path = cfg_mod.CONFIG_PATH
        try:
            cfg_mod.CONFIG_DIR = Path(tmpdir) / ".deep-research"
            cfg_mod.CONFIG_PATH = cfg_mod.CONFIG_DIR / "config.yaml"
            cfg_mod.save_config(cfg)
            assert cfg_mod.CONFIG_PATH.exists()

            # Clear cache and reload
            cfg_mod._config = None
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
                reloaded = cfg_mod.load_config()
                assert reloaded.default_provider == "openai"
                assert reloaded.default_model == "gpt-4o"
                assert reloaded.max_iterations == 5
        finally:
            cfg_mod.CONFIG_DIR = original_dir
            cfg_mod.CONFIG_PATH = original_path


# ── get_config cache behavior ──────────────────────────────────

def test_get_config_caches_result():
    from src.backend import config
    config._config = None
    with patch("src.backend.config.load_config") as mock_load:
        mock_load.return_value = config.AppConfig(providers={})
        config.get_config()
        config.get_config()
        assert mock_load.call_count == 1


# ── reload_config forces reload ────────────────────────────────

def test_reload_config_refreshes():
    from src.backend import config
    config._config = None
    with patch("src.backend.config.load_config") as mock_load:
        mock_load.return_value = config.AppConfig(providers={})
        config.get_config()
        config.reload_config()
        assert mock_load.call_count == 2


# ── ProviderConfig and RoleConfig dataclasses ──────────────────

def test_provider_config_defaults():
    from src.backend.config import ProviderConfig
    pc = ProviderConfig(name="test", type="openai")
    assert pc.name == "test"
    assert pc.base_url == ""
    assert pc.api_key == ""


def test_role_config_defaults():
    from src.backend.config import RoleConfig
    rc = RoleConfig(provider="openai", model="gpt-4o")
    assert rc.temperature == 0.2  # default

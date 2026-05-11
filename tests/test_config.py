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


def test_load_config_defaults():
    with patch("src.backend.config._load_yaml_config", return_value={}), \
         patch.dict(os.environ, {}, clear=True):
        from src.backend.config import load_config
        cfg = load_config()
        assert cfg.base_url == "https://api.openai.com/v1"
        assert cfg.api_key == ""
        assert cfg.default_model == "gpt-4o"
        assert cfg.quality_threshold == 0.6


def test_load_config_from_yaml():
    with patch("src.backend.config._load_yaml_config", return_value={
        "llm": {"base_url": "https://custom.api/v1", "api_key": "sk-test", "model": "custom-model"}
    }), patch.dict(os.environ, {}, clear=True):
        from src.backend.config import load_config
        cfg = load_config()
        assert cfg.base_url == "https://custom.api/v1"
        assert cfg.api_key == "sk-test"
        assert cfg.default_model == "custom-model"


def test_role_override_from_env():
    with patch("src.backend.config._load_yaml_config", return_value={}), \
         patch.dict(os.environ, {
             "PLANNER_MODEL": "gpt-4o-pro",
         }, clear=True):
        from src.backend.config import load_config
        cfg = load_config()
        assert "planner" in cfg.roles
        assert cfg.roles["planner"].model == "gpt-4o-pro"


def test_get_role_falls_back_to_default():
    with patch("src.backend.config._load_yaml_config", return_value={}), \
         patch.dict(os.environ, {}, clear=True):
        from src.backend.config import load_config
        cfg = load_config()
        role = cfg.get_role("subagent")
        assert role.model == cfg.default_model
        assert role.temperature == 0.2


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


# ── API key / base_url / model priority: env > yaml ────────────

def test_api_key_env_wins():
    with patch("src.backend.config._load_yaml_config", return_value={
        "llm": {"api_key": "yaml-key"}
    }), patch.dict(os.environ, {"LLM_API_KEY": "env-key"}, clear=True):
        from src.backend.config import load_config
        cfg = load_config()
        assert cfg.api_key == "env-key"


def test_base_url_env_wins():
    with patch("src.backend.config._load_yaml_config", return_value={
        "llm": {"base_url": "https://yaml.api/v1"}
    }), patch.dict(os.environ, {"LLM_BASE_URL": "https://env.api/v1"}, clear=True):
        from src.backend.config import load_config
        cfg = load_config()
        assert cfg.base_url == "https://env.api/v1"


def test_model_env_wins():
    with patch("src.backend.config._load_yaml_config", return_value={
        "llm": {"model": "yaml-model"}
    }), patch.dict(os.environ, {"LLM_MODEL": "env-model"}, clear=True):
        from src.backend.config import load_config
        cfg = load_config()
        assert cfg.default_model == "env-model"


# ── Research params: env > yaml > builtin ──────────────────────

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
        assert cfg.quality_threshold == 0.6
        assert cfg.max_sources_per_domain == 5


# ── save_config + reload_config round-trip ─────────────────────

def test_save_and_reload_roundtrip():
    from src.backend import config as cfg_mod
    import yaml

    cfg = cfg_mod.AppConfig(
        base_url="https://test.api/v1",
        api_key="sk-test",
        default_model="gpt-4o",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = cfg_mod.CONFIG_DIR
        original_path = cfg_mod.CONFIG_PATH
        try:
            cfg_mod.CONFIG_DIR = Path(tmpdir) / ".deep-research"
            cfg_mod.CONFIG_PATH = cfg_mod.CONFIG_DIR / "config.yaml"
            cfg_mod.save_config(cfg)
            assert cfg_mod.CONFIG_PATH.exists()

            cfg_mod._config = None
            reloaded = cfg_mod.load_config()
            assert reloaded.base_url == "https://test.api/v1"
            assert reloaded.api_key == "sk-test"
            assert reloaded.default_model == "gpt-4o"
        finally:
            cfg_mod.CONFIG_DIR = original_dir
            cfg_mod.CONFIG_PATH = original_path


# ── get_config cache behavior ──────────────────────────────────

def test_get_config_caches_result():
    from src.backend import config
    config._config = None
    with patch("src.backend.config.load_config") as mock_load:
        mock_load.return_value = config.AppConfig()
        config.get_config()
        config.get_config()
        assert mock_load.call_count == 1


# ── reload_config forces reload ────────────────────────────────

def test_reload_config_refreshes():
    from src.backend import config
    config._config = None
    with patch("src.backend.config.load_config") as mock_load:
        mock_load.return_value = config.AppConfig()
        config.get_config()
        config.reload_config()
        assert mock_load.call_count == 2


# ── RoleConfig dataclass ───────────────────────────────────────

def test_role_config_defaults():
    from src.backend.config import RoleConfig
    rc = RoleConfig(model="gpt-4o")
    assert rc.temperature == 0.2  # default
    assert rc.model == "gpt-4o"

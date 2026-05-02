"""Tests for provider layer."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def test_get_provider_openai_type():
    from src.backend.providers import get_provider

    with patch("src.backend.providers.openai_compatible.AsyncOpenAI"):
        p = get_provider("openai", "https://api.example.com/v1", "sk-test")
        assert p.name == "openai_compatible"


def test_get_provider_anthropic_type():
    from src.backend.providers import get_provider

    with patch("src.backend.providers.anthropic.anthropic.AsyncAnthropic"):
        p = get_provider("anthropic", "", "sk-test")
        assert p.name == "anthropic"


def test_get_provider_unknown_type():
    from src.backend.providers import get_provider

    with pytest.raises(ValueError, match="Unknown provider type"):
        get_provider("unknown", "", "")


def test_get_provider_with_extra_headers():
    from src.backend.providers import get_provider

    with patch("src.backend.providers.openai_compatible.AsyncOpenAI"):
        p = get_provider("openai", "https://api.example.com/v1", "sk-test", extra_headers={"X-Custom": "value"})
        assert p._extra_headers == {"X-Custom": "value"}


# ── Retry logic tests ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_chat_with_retry_success_on_first_try():
    from src.backend.providers.base import LLMProvider
    import asyncio

    class TestProvider(LLMProvider):
        name = "test"

        async def chat(self, model, messages, temperature=0.2, max_tokens=None):
            return "success"

    provider = TestProvider()
    result = await provider.chat_with_retry(
        model="test-model",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert result == "success"


@pytest.mark.asyncio
async def test_chat_with_retry_transient_error():
    from src.backend.providers.base import LLMProvider
    import asyncio

    call_count = [0]

    class TestProvider(LLMProvider):
        name = "test"

        async def chat(self, model, messages, temperature=0.2, max_tokens=None):
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("rate limit exceeded")
            return "success"

    provider = TestProvider()
    result = await provider.chat_with_retry(
        model="test-model",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert result == "success"
    assert call_count[0] == 3


@pytest.mark.asyncio
async def test_chat_with_retry_fatal_error_no_retry():
    from src.backend.providers.base import LLMProvider
    import asyncio

    class TestProvider(LLMProvider):
        name = "test"

        async def chat(self, model, messages, temperature=0.2, max_tokens=None):
            raise Exception("401 Unauthorized")

    provider = TestProvider()
    with pytest.raises(Exception, match="401 Unauthorized"):
        await provider.chat_with_retry(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
        )


@pytest.mark.asyncio
async def test_chat_with_retry_exhausted():
    from src.backend.providers.base import LLMProvider
    import asyncio

    class TestProvider(LLMProvider):
        name = "test"

        async def chat(self, model, messages, temperature=0.2, max_tokens=None):
            raise RuntimeError("always fails")

    provider = TestProvider()
    with pytest.raises(RuntimeError, match="always fails"):
        await provider.chat_with_retry(
            model="test-model",
            messages=[{"role": "user", "content": "Hi"}],
            max_retries=3,
        )


def test_is_fatal_auth_errors():
    from src.backend.providers.base import _is_fatal

    assert _is_fatal(Exception("401 Unauthorized")) is True
    assert _is_fatal(Exception("403 Forbidden")) is True
    assert _is_fatal(Exception("invalid API key")) is True
    assert _is_fatal(Exception("authentication failed")) is True
    assert _is_fatal(Exception("not found")) is True


def test_is_fatal_transient_errors():
    from src.backend.providers.base import _is_fatal

    assert _is_fatal(Exception("rate limit exceeded")) is False
    assert _is_fatal(Exception("429 too many requests")) is False
    assert _is_fatal(Exception("server error")) is False
    assert _is_fatal(Exception("timeout")) is False

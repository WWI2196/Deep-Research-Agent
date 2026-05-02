"""Tests for OpenAI-compatible provider chat method."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_openai_compatible_chat_basic():
    from src.backend.providers.openai_compatible import OpenAICompatibleProvider

    mock_client = AsyncMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Hello, world!"
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    with patch("src.backend.providers.openai_compatible.AsyncOpenAI", return_value=mock_client):
        provider = OpenAICompatibleProvider("https://api.example.com/v1", "sk-test")
        result = await provider.chat(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result == "Hello, world!"
        mock_client.chat.completions.create.assert_called_once()
        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["temperature"] == 0.2


@pytest.mark.asyncio
async def test_openai_compatible_chat_with_max_tokens():
    from src.backend.providers.openai_compatible import OpenAICompatibleProvider

    mock_client = AsyncMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Response"
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    with patch("src.backend.providers.openai_compatible.AsyncOpenAI", return_value=mock_client):
        provider = OpenAICompatibleProvider("https://api.example.com/v1", "sk-test")
        result = await provider.chat(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1024,
        )
        assert result == "Response"
        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["max_tokens"] == 1024


@pytest.mark.asyncio
async def test_openai_compatible_chat_empty_content():
    from src.backend.providers.openai_compatible import OpenAICompatibleProvider

    mock_client = AsyncMock()
    mock_choice = MagicMock()
    mock_choice.message.content = ""
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    with patch("src.backend.providers.openai_compatible.AsyncOpenAI", return_value=mock_client):
        provider = OpenAICompatibleProvider("https://api.example.com/v1", "sk-test")
        result = await provider.chat(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result == ""


@pytest.mark.asyncio
async def test_openai_compatible_chat_with_none_content():
    from src.backend.providers.openai_compatible import OpenAICompatibleProvider

    mock_client = AsyncMock()
    mock_choice = MagicMock()
    mock_choice.message.content = None
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    with patch("src.backend.providers.openai_compatible.AsyncOpenAI", return_value=mock_client):
        provider = OpenAICompatibleProvider("https://api.example.com/v1", "sk-test")
        result = await provider.chat(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result == ""

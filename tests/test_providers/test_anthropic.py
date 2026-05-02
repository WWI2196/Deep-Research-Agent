"""Tests for Anthropic provider chat method."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_anthropic_chat_basic():
    from src.backend.providers.anthropic import AnthropicProvider

    mock_client = AsyncMock()
    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = "Hello from Claude"
    mock_client.messages.create.return_value.content = [mock_text_block]

    with patch("src.backend.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
        provider = AnthropicProvider("sk-test")
        result = await provider.chat(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result == "Hello from Claude"
        mock_client.messages.create.assert_called_once()
        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_anthropic_chat_system_message_separation():
    from src.backend.providers.anthropic import AnthropicProvider

    mock_client = AsyncMock()
    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = "Response"
    mock_client.messages.create.return_value.content = [mock_text_block]

    with patch("src.backend.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
        provider = AnthropicProvider("sk-test")
        await provider.chat(
            model="claude-sonnet-4-6",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        )
        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs["system"] == "You are helpful."
        assert len(kwargs["messages"]) == 1
        assert kwargs["messages"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_anthropic_chat_multiple_system_messages():
    from src.backend.providers.anthropic import AnthropicProvider

    mock_client = AsyncMock()
    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = "Response"
    mock_client.messages.create.return_value.content = [mock_text_block]

    with patch("src.backend.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
        provider = AnthropicProvider("sk-test")
        await provider.chat(
            model="claude-sonnet-4-6",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"},
            ],
        )
        kwargs = mock_client.messages.create.call_args.kwargs
        assert "You are helpful.\n\nBe concise." in kwargs["system"]


@pytest.mark.asyncio
async def test_anthropic_chat_all_system_no_user():
    from src.backend.providers.anthropic import AnthropicProvider

    mock_client = AsyncMock()
    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = "Response"
    mock_client.messages.create.return_value.content = [mock_text_block]

    with patch("src.backend.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
        provider = AnthropicProvider("sk-test")
        await provider.chat(
            model="claude-sonnet-4-6",
            messages=[
                {"role": "system", "content": "You are helpful."},
            ],
        )
        kwargs = mock_client.messages.create.call_args.kwargs
        # Should fallback to a dummy user message
        assert kwargs["messages"][0]["role"] == "user"
        assert kwargs["messages"][0]["content"] == "Please respond."


@pytest.mark.asyncio
async def test_anthropic_chat_multiple_text_blocks():
    from src.backend.providers.anthropic import AnthropicProvider

    mock_client = AsyncMock()
    block1 = MagicMock()
    block1.type = "text"
    block1.text = "First part. "
    block2 = MagicMock()
    block2.type = "text"
    block2.text = "Second part."
    mock_client.messages.create.return_value.content = [block1, block2]

    with patch("src.backend.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
        provider = AnthropicProvider("sk-test")
        result = await provider.chat(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result == "First part. Second part."


@pytest.mark.asyncio
async def test_anthropic_chat_custom_max_tokens():
    from src.backend.providers.anthropic import AnthropicProvider

    mock_client = AsyncMock()
    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = "Response"
    mock_client.messages.create.return_value.content = [mock_text_block]

    with patch("src.backend.providers.anthropic.anthropic.AsyncAnthropic", return_value=mock_client):
        provider = AnthropicProvider("sk-test")
        await provider.chat(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=2048,
        )
        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs["max_tokens"] == 2048

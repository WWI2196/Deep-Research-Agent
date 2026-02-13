"""Anthropic Claude provider.

The Anthropic Messages API differs from OpenAI:
  - System message is a top-level parameter, not in the messages array.
  - Only "user" and "assistant" roles allowed in messages.

This provider transparently adapts OpenAI-style messages.

Requires: ANTHROPIC_API_KEY environment variable.
"""

import os
from typing import Any, Dict, List, Optional

import anthropic

from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    DEFAULT_MODELS = [
        "claude-sonnet-4-20250514",
        "claude-haiku-3-5-20241022",
    ]

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required. "
                "Get one at https://console.anthropic.com/settings/keys"
            )
        self.client = anthropic.Anthropic(api_key=api_key)

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        # Separate system messages from conversation messages
        system_parts: List[str] = []
        conversation: List[Dict[str, str]] = []

        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                conversation.append({"role": msg["role"], "content": msg["content"]})

        if not conversation:
            conversation = [{"role": "user", "content": "Please respond."}]

        # Hint JSON mode via system prompt
        if response_format and response_format.get("type") == "json_object":
            system_parts.append("Respond with valid JSON only.")

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": conversation,
            "temperature": temperature,
            "max_tokens": 8192,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        response = self.client.messages.create(**kwargs)

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text
        return text

    def get_available_models(self) -> List[str]:
        return self.DEFAULT_MODELS

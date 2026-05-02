"""Anthropic native provider — uses the Messages API directly."""

import anthropic

from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        system_parts: list[str] = []
        conversation: list[dict[str, str]] = []

        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                conversation.append({"role": msg["role"], "content": msg["content"]})

        if not conversation:
            conversation = [{"role": "user", "content": "Please respond."}]

        response = await self.client.messages.create(
            model=model,
            system="\n\n".join(system_parts) if system_parts else anthropic.NOT_GIVEN,
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens or 8192,
        )

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text
        return text

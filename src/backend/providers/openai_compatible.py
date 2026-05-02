"""Generic OpenAI-compatible provider — works with any API that follows the OpenAI chat completions format (OpenAI, Gemini, OpenRouter, DeepSeek, Qwen, etc.)."""

from openai import AsyncOpenAI

from .base import LLMProvider


class OpenAICompatibleProvider(LLMProvider):
    name = "openai_compatible"

    def __init__(self, base_url: str, api_key: str) -> None:
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        kwargs: dict = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

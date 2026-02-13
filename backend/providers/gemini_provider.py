"""Google Gemini provider via OpenAI-compatible endpoint.

Gemini natively supports the OpenAI chat completions format:
  base_url = https://generativelanguage.googleapis.com/v1beta/openai/

Requires: GEMINI_API_KEY environment variable.
"""

import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .base import LLMProvider


class GeminiProvider(LLMProvider):
    name = "gemini"

    DEFAULT_MODELS = [
        "gemini-3-pro-preview",
        "gemini-2.5-pro",

    ]

    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is required. "
                "Get one at https://aistudio.google.com/apikey"
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def get_available_models(self) -> List[str]:
        return self.DEFAULT_MODELS

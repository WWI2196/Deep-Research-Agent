"""OpenAI provider (GPT-4o, o1, etc.).

Requires: OPENAI_API_KEY environment variable.
"""

import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    name = "openai"

    DEFAULT_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "o3-mini",
    ]

    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required. "
                "Get one at https://platform.openai.com/api-keys"
            )
        self.client = OpenAI(api_key=api_key)

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def get_available_models(self) -> List[str]:
        return self.DEFAULT_MODELS

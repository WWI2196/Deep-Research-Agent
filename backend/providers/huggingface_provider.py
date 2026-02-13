"""HuggingFace Inference API provider.

Supports multiple HF inference providers (novita, sambanova, auto, etc.).
Handles DeepSeek <think> tag cleaning.

Requires: HF_TOKEN environment variable.
"""

import os
import re
from typing import Any, Dict, List, Optional

from huggingface_hub import InferenceClient

from .base import LLMProvider


class HuggingFaceProvider(LLMProvider):
    name = "huggingface"

    DEFAULT_MODELS = [
        "Qwen/Qwen2.5-72B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "MiniMaxAI/MiniMax-M1-80k",
    ]

    def __init__(self, hf_provider: str = "novita"):
        api_key = os.environ.get("HF_TOKEN", "")
        if not api_key:
            raise ValueError(
                "HF_TOKEN is required. "
                "Get one at https://huggingface.co/settings/tokens"
            )
        self.hf_provider = hf_provider
        self.client = InferenceClient(
            api_key=api_key,
            provider=hf_provider,
        )

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

        completion = self.client.chat.completions.create(**kwargs)
        content = completion.choices[0].message.content or ""

        # Strip DeepSeek <think> blocks
        if "<think>" in content and "</think>" in content:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content

    def get_available_models(self) -> List[str]:
        return self.DEFAULT_MODELS

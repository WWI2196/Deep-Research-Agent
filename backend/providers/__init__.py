"""LLM Provider registry.

Supported providers:
  - gemini      (Google Gemini via OpenAI-compatible endpoint)
  - openai      (OpenAI GPT models)
  - anthropic   (Anthropic Claude models)
  - huggingface (HuggingFace Inference API)
"""

from typing import Dict

from .base import LLMProvider

_cache: Dict[str, LLMProvider] = {}


def get_provider(name: str, **kwargs) -> LLMProvider:
    """Get or create a cached provider instance by name."""
    cache_key = f"{name}:{hash(frozenset(kwargs.items()))}" if kwargs else name

    if cache_key not in _cache:
        if name == "gemini":
            from .gemini_provider import GeminiProvider
            _cache[cache_key] = GeminiProvider(**kwargs)
        elif name == "openai":
            from .openai_provider import OpenAIProvider
            _cache[cache_key] = OpenAIProvider(**kwargs)
        elif name == "anthropic":
            from .anthropic_provider import AnthropicProvider
            _cache[cache_key] = AnthropicProvider(**kwargs)
        elif name == "huggingface":
            from .huggingface_provider import HuggingFaceProvider
            _cache[cache_key] = HuggingFaceProvider(**kwargs)
        else:
            raise ValueError(
                f"Unknown provider: '{name}'. "
                f"Supported: gemini, openai, anthropic, huggingface"
            )
    return _cache[cache_key]


def clear_cache():
    """Clear provider cache (useful after config changes)."""
    _cache.clear()


def list_providers() -> list:
    return ["gemini", "openai", "anthropic", "huggingface"]


__all__ = ["LLMProvider", "get_provider", "list_providers", "clear_cache"]

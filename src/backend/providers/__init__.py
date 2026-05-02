"""LLM Provider registry — two generic modes, user-configured."""

from .base import LLMProvider


def get_provider(provider_type: str, base_url: str, api_key: str, *, extra_headers: dict | None = None) -> LLMProvider:
    """Create a provider instance.

    provider_type: "openai" for OpenAI-compatible APIs, "anthropic" for native Anthropic
    base_url: API base URL (openai type only, ignored for anthropic)
    api_key: API key for the provider
    """
    if provider_type == "openai":
        from .openai_compatible import OpenAICompatibleProvider
        p = OpenAICompatibleProvider(base_url, api_key)
        if extra_headers:
            p._extra_headers = extra_headers
        return p
    elif provider_type == "anthropic":
        from .anthropic import AnthropicProvider
        return AnthropicProvider(api_key)
    else:
        raise ValueError(f"Unknown provider type: '{provider_type}'. Must be 'openai' or 'anthropic'.")


__all__ = ["LLMProvider", "get_provider"]

"""Unified async LLM call routing with provider caching."""

import asyncio
import datetime
import logging
import time

from .config import get_config
from .helpers import clean_think_tags
from .providers import get_provider
from .tracing import trace_llm_call

logger = logging.getLogger(__name__)

# ── provider cache with config-aware TTL ──────────────────────────────

_provider_cache: dict[str, tuple[object, float]] = {}
_PROVIDER_CACHE_TTL = 300  # 5 minutes


def _get_or_create_provider(provider_name: str):
    now = time.monotonic()
    entry = _provider_cache.get(provider_name)
    if entry is not None:
        cached_provider, cached_at = entry
        if now - cached_at < _PROVIDER_CACHE_TTL:
            return cached_provider

    app_cfg = get_config()
    pc = app_cfg.providers.get(provider_name)
    if not pc:
        pc = app_cfg.providers.get(app_cfg.default_provider)
    if not pc:
        raise RuntimeError(f"No provider configured: '{provider_name}'")

    p = get_provider(pc.type, pc.base_url, pc.api_key)
    _provider_cache[provider_name] = (p, now)
    return p


def invalidate_provider_cache() -> None:
    """Clear cached providers so next call picks up config changes."""
    _provider_cache.clear()


# ── unified async LLM call ────────────────────────────────────────────

async def chat(
    role: str,
    messages: list[dict[str, str]],
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_retries: int = 3,
) -> str:
    role_cfg = get_config().get_role(role)
    provider = _get_or_create_provider(role_cfg.provider)
    temp = temperature if temperature is not None else role_cfg.temperature

    today = datetime.date.today().isoformat()
    date_prefix = f"Today's date: {today}. Use this as the current date when reasoning about time-sensitive topics."
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = messages[0]["content"] + "\n\n" + date_prefix
    else:
        messages = [{"role": "system", "content": date_prefix}, *messages]

    for attempt in range(max_retries):
        start = time.monotonic()
        try:
            result = await provider.chat(
                model=role_cfg.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens,
            )
            result = clean_think_tags(result)
            latency_ms = int((time.monotonic() - start) * 1000)
            await trace_llm_call(
                role=role,
                messages=messages,
                provider=role_cfg.provider,
                model=role_cfg.model,
                response=result,
                temperature=temp,
                max_tokens=max_tokens,
                latency_ms=latency_ms,
                retry_attempt=attempt,
            )
            return result
        except Exception as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            err = str(exc).lower()
            if any(c in err for c in ["401", "403", "invalid"]):
                await trace_llm_call(
                    role=role,
                    messages=messages,
                    provider=role_cfg.provider,
                    model=role_cfg.model,
                    temperature=temp,
                    max_tokens=max_tokens,
                    latency_ms=latency_ms,
                    retry_attempt=attempt,
                    error=str(exc),
                )
                raise
            if attempt < max_retries - 1:
                delay = 2.0 * (2**attempt)
                logger.warning("LLM call [%s] attempt %d failed: %s. Retrying in %.1fs", role, attempt + 1, exc, delay)
                await trace_llm_call(
                    role=role,
                    messages=messages,
                    provider=role_cfg.provider,
                    model=role_cfg.model,
                    temperature=temp,
                    max_tokens=max_tokens,
                    latency_ms=latency_ms,
                    retry_attempt=attempt,
                    error=str(exc),
                )
                await asyncio.sleep(delay)
                continue
            await trace_llm_call(
                role=role,
                messages=messages,
                provider=role_cfg.provider,
                model=role_cfg.model,
                temperature=temp,
                max_tokens=max_tokens,
                latency_ms=latency_ms,
                retry_attempt=attempt,
                error=str(exc),
            )
            raise

    raise RuntimeError(f"LLM call exhausted retries for [{role}]")

"""Unified async LLM call routing with OpenAI-compatible API."""

import asyncio
import datetime
import logging
import time

from openai import AsyncOpenAI

from .config import get_config
from .helpers import strip_llm_artifacts
from .tracing import trace_llm_call

logger = logging.getLogger(__name__)

# ── single client cache ───────────────────────────────────────────────

_client: AsyncOpenAI | None = None
_client_cfg_key: str = ""


def _get_client() -> AsyncOpenAI:
    global _client, _client_cfg_key
    cfg = get_config()
    key = f"{cfg.base_url}:{cfg.api_key}"
    if _client is None or _client_cfg_key != key:
        _client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        _client_cfg_key = key
    return _client


def invalidate_client_cache() -> None:
    """Clear cached client so next call picks up config changes."""
    global _client, _client_cfg_key
    _client = None
    _client_cfg_key = ""


# ── unified async LLM call ────────────────────────────────────────────

async def chat(
    role: str,
    messages: list[dict[str, str]],
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_retries: int = 3,
) -> str:
    cfg = get_config()
    role_cfg = cfg.get_role(role)
    client = _get_client()
    temp = temperature if temperature is not None else role_cfg.temperature
    model = role_cfg.model

    today = datetime.date.today().isoformat()
    date_prefix = f"Today's date: {today}. Use this as the current date when reasoning about time-sensitive topics."
    # Shallow-copy messages to avoid mutating the caller's list
    messages = [dict(m) for m in messages]
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = messages[0]["content"] + "\n\n" + date_prefix
    else:
        messages = [{"role": "system", "content": date_prefix}, *messages]

    for attempt in range(max_retries):
        start = time.monotonic()
        try:
            kwargs: dict = {"model": model, "messages": messages, "temperature": temp}
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            response = await client.chat.completions.create(**kwargs)
            result = response.choices[0].message.content or ""
            result = strip_llm_artifacts(result)
            latency_ms = int((time.monotonic() - start) * 1000)
            await trace_llm_call(
                role=role,
                messages=messages,
                provider=cfg.base_url,
                model=model,
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
                    provider=cfg.base_url,
                    model=model,
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
                    provider=cfg.base_url,
                    model=model,
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
                provider=cfg.base_url,
                model=model,
                temperature=temp,
                max_tokens=max_tokens,
                latency_ms=latency_ms,
                retry_attempt=attempt,
                error=str(exc),
            )
            raise

    raise RuntimeError(f"LLM call exhausted retries for [{role}]")

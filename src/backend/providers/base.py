"""Abstract base class for LLM providers."""

import asyncio
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str: ...

    async def chat_with_retry(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        max_retries: int = 3,
    ) -> str:
        """Send a chat request with exponential backoff on transient errors."""
        for attempt in range(max_retries):
            try:
                return await self.chat(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                if _is_fatal(exc):
                    raise
                if attempt == max_retries - 1:
                    raise
                delay = 2.0 * (2**attempt)
                logger.warning(
                    "Provider %s attempt %d failed: %s. Retrying in %.1fs...",
                    self.name, attempt + 1, exc, delay,
                )
                await asyncio.sleep(delay)
        raise RuntimeError("unreachable")


def _is_fatal(exc: Exception) -> bool:
    msg = str(exc).lower()
    fatal = {"401", "403", "invalid", "auth", "unauthorized", "not found"}
    return any(k in msg for k in fatal)

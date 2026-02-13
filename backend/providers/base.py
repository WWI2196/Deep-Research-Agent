"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMProvider(ABC):
    """Base class for all LLM providers.

    Every provider accepts OpenAI-style messages:
        [{"role": "system"|"user"|"assistant", "content": "..."}]
    and returns a plain-text string.
    """

    name: str = "base"

    @abstractmethod
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send chat messages and return the text response."""
        ...

    def get_available_models(self) -> List[str]:
        """Return commonly used model IDs for this provider."""
        return []

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

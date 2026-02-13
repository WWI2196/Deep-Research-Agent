"""Firecrawl search and content-extraction helpers."""

import os
import time
from typing import Any, Dict, List

from firecrawl import Firecrawl

FIRECRAWL_MAX_RETRIES = int(os.environ.get("FIRECRAWL_MAX_RETRIES", "3"))
FIRECRAWL_BACKOFF_BASE = float(os.environ.get("FIRECRAWL_BACKOFF_BASE", "0.5"))


def _get_firecrawl() -> Firecrawl:
    return Firecrawl(api_key=os.environ.get("FIRECRAWL_API_KEY", ""))


def _retry(func, *args, **kwargs):
    last_err = None
    for attempt in range(FIRECRAWL_MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_err = exc
            time.sleep(FIRECRAWL_BACKOFF_BASE * (2 ** attempt))
    raise last_err  # type: ignore


def search(query: str, limit: int = 5) -> Dict[str, Any]:
    """Run a Firecrawl web search with retries."""
    fc = _get_firecrawl()
    return _retry(fc.search, query=query, limit=limit)


def extract(urls: List[str], prompt: str) -> Dict[str, Any]:
    """Extract structured content from URLs via Firecrawl."""
    fc = _get_firecrawl()
    return _retry(fc.extract, urls, {"prompt": prompt})

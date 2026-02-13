"""Firecrawl search and content-extraction helpers.

Firecrawl v4 SDK returns Pydantic models (SearchData, etc.) instead of
plain dicts.  The helpers below normalise every response into the
``{"data": [...]}`` shape that the rest of the pipeline expects.
"""

import logging
import os
import time
from typing import Any, Dict, List

from firecrawl import Firecrawl

logger = logging.getLogger(__name__)

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


# ── response normalisers ────────────────────────────────────────────

def _pydantic_to_dict(obj: Any) -> Any:
    """Recursively convert a Pydantic model (or list of them) to dicts."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _pydantic_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_pydantic_to_dict(item) for item in obj]
    # Pydantic v2 model
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    # Pydantic v1 model
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def _normalise_search_response(result: Any) -> Dict[str, Any]:
    """Turn a Firecrawl v4 SearchData into ``{"data": [...]}``.

    SearchData has ``.web``, ``.news``, ``.images`` — each is a list
    of ``SearchResultWeb`` / ``SearchResultNews`` / ``Document`` objects.
    We flatten all non-None lists into one ``data`` list of plain dicts.
    """
    if result is None:
        return {"data": []}

    # Already a dict (older SDK or pre-normalised)
    if isinstance(result, dict):
        return result

    items: List[Dict[str, Any]] = []

    # Pydantic model with typed lists
    for attr in ("web", "news", "images"):
        bucket = getattr(result, attr, None)
        if bucket and isinstance(bucket, list):
            for entry in bucket:
                d = _pydantic_to_dict(entry)
                if isinstance(d, dict):
                    # Ensure required fields for downstream _normalize_search_item
                    if d.get("metadata") and isinstance(d["metadata"], dict):
                        meta = d["metadata"]
                        if not d.get("url") and meta.get("sourceURL"):
                            d["url"] = meta["sourceURL"]
                        if not d.get("title") and meta.get("title"):
                            d["title"] = meta["title"]
                        if not d.get("description") and meta.get("description"):
                            d["description"] = meta["description"]
                    items.append(d)

    if not items:
        # Last-ditch: try model_dump and look for any list-valued key
        try:
            dumped = _pydantic_to_dict(result)
            if isinstance(dumped, dict):
                for v in dumped.values():
                    if isinstance(v, list):
                        items.extend(
                            e for e in v if isinstance(e, dict)
                        )
        except Exception:
            pass

    logger.debug("Normalised search → %d items", len(items))
    return {"data": items}


def _normalise_extract_response(result: Any) -> Dict[str, Any]:
    """Turn a Firecrawl v4 extract response into ``{"data": ...}``."""
    if result is None:
        return {"data": None}
    if isinstance(result, dict):
        return result
    if isinstance(result, (str, list)):
        return {"data": result}

    # Pydantic model
    dumped = _pydantic_to_dict(result)
    if isinstance(dumped, dict):
        # If the dump already has a 'data' key, keep it
        if "data" in dumped:
            return dumped
        return {"data": dumped}

    return {"data": dumped}


# ── public API ──────────────────────────────────────────────────────

def search(query: str, limit: int = 5) -> Dict[str, Any]:
    """Run a Firecrawl web search with retries.

    Always returns ``{"data": [<dict>, ...]}``.
    """
    fc = _get_firecrawl()
    raw = _retry(fc.search, query=query, limit=limit)
    return _normalise_search_response(raw)


def extract(urls: List[str], prompt: str) -> Dict[str, Any]:
    """Extract structured content from URLs via Firecrawl.

    Always returns a plain dict.
    """
    fc = _get_firecrawl()
    raw = _retry(fc.extract, urls, prompt=prompt)
    return _normalise_extract_response(raw)

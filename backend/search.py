"""Search and content-extraction helpers.

Primary: Firecrawl (v4 SDK – returns Pydantic models).
Fallback: DuckDuckGo (free, no API key required).

Every public function returns the ``{"data": [...]}`` shape
that the rest of the pipeline expects.
"""

import logging
import os
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

FIRECRAWL_MAX_RETRIES = int(os.environ.get("FIRECRAWL_MAX_RETRIES", "3"))
FIRECRAWL_BACKOFF_BASE = float(os.environ.get("FIRECRAWL_BACKOFF_BASE", "0.5"))

# Track whether Firecrawl is usable this session (avoids repeated failures)
_firecrawl_disabled = False


def _get_firecrawl():
    """Return a Firecrawl client, or None if unavailable."""
    try:
        from firecrawl import Firecrawl
    except ImportError:
        return None
    api_key = os.environ.get("FIRECRAWL_API_KEY", "")
    if not api_key:
        return None
    return Firecrawl(api_key=api_key)


def _retry(func, *args, **kwargs):
    last_err = None
    for attempt in range(FIRECRAWL_MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_err = exc
            err_str = str(exc).lower()
            # Don't retry payment / auth errors
            if any(k in err_str for k in ("payment", "402", "401", "403", "insufficient", "unauthorized")):
                raise
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

def _ddg_search(query: str, limit: int = 5) -> Dict[str, Any]:
    """Free DuckDuckGo fallback – no API key needed."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning("Neither ddgs nor duckduckgo_search installed; returning empty results")
            return {"data": []}

    items: List[Dict[str, Any]] = []
    try:
        results = list(DDGS(timeout=10).text(query, max_results=limit))
        for r in results:
            items.append({
                "title": r.get("title", ""),
                "url": r.get("href", r.get("link", "")),
                "description": r.get("body", r.get("snippet", "")),
            })
    except Exception as exc:
        logger.warning("DuckDuckGo search failed: %s", exc)
    logger.info("DuckDuckGo → %d items for '%s'", len(items), query[:60])
    return {"data": items}


def search(query: str, limit: int = 5) -> Dict[str, Any]:
    """Run a web search with retries.

    Tries Firecrawl first; if it fails (credits, auth, missing key),
    falls back to DuckDuckGo for free.

    Always returns ``{"data": [<dict>, ...]}``.
    """
    global _firecrawl_disabled

    # ── try Firecrawl ──
    if not _firecrawl_disabled:
        fc = _get_firecrawl()
        if fc is not None:
            try:
                raw = _retry(fc.search, query=query, limit=limit)
                result = _normalise_search_response(raw)
                if result.get("data"):
                    return result
                # Firecrawl returned empty data – try DDG
                logger.info("Firecrawl returned 0 items for '%s', trying DuckDuckGo", query)
            except Exception as exc:
                err = str(exc).lower()
                if any(k in err for k in ("payment", "402", "insufficient", "credit")):
                    logger.warning("Firecrawl credits exhausted – disabling for this session. Using DuckDuckGo.")
                    _firecrawl_disabled = True
                else:
                    logger.warning("Firecrawl search error: %s – falling back to DuckDuckGo", exc)

    # ── fallback: DuckDuckGo ──
    return _ddg_search(query, limit=limit)


def extract(urls: List[str], prompt: str) -> Dict[str, Any]:
    """Extract structured content from URLs via Firecrawl.

    Falls back gracefully if Firecrawl is unavailable.
    Always returns a plain dict.
    """
    if _firecrawl_disabled:
        return {"data": None}

    fc = _get_firecrawl()
    if fc is None:
        return {"data": None}

    try:
        raw = _retry(fc.extract, urls, prompt=prompt)
        return _normalise_extract_response(raw)
    except Exception as exc:
        logger.warning("Firecrawl extract failed: %s", exc)
        return {"data": None}

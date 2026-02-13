"""Thread-safe event bus for real-time observability.

Pipeline components emit events from background threads.
The SSE server subscribes to propagate them to the frontend.
"""

import threading
import time
from typing import Callable, List

_listeners: List[Callable[[dict], None]] = []
_lock = threading.Lock()


def add_listener(fn: Callable[[dict], None]):
    """Subscribe to all pipeline events."""
    with _lock:
        _listeners.append(fn)


def remove_listener(fn: Callable[[dict], None]):
    """Unsubscribe."""
    with _lock:
        try:
            _listeners.remove(fn)
        except ValueError:
            pass


def emit(event: dict):
    """Broadcast an event to every registered listener."""
    event.setdefault("timestamp", time.time())
    with _lock:
        for fn in _listeners:
            try:
                fn(event)
            except Exception:
                pass

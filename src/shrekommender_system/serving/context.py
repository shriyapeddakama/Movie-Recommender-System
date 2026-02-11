"""Utilities for building request level context for routing decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RequestContext:
    """Normalized view of the incoming request.

    The router only depends on this lightweight structure so that
    additional request metadata can be plugged in without touching the
    router itself.
    """

    user_id: str
    traits: Dict[str, Any] = field(default_factory=dict)
    request_metadata: Dict[str, Any] = field(default_factory=dict)
    model_hint: Optional[str] = None

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Retrieve nested values using dotted keys (e.g. ``user.segment``)."""

        current: Any = {
            "user": {"id": self.user_id, **self.traits},
            "request": self.request_metadata,
        }
        for part in dotted_key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current


def build_context(user_id: str, traits: Optional[Dict[str, Any]] = None, **metadata: Any) -> RequestContext:
    """Convenience helper used by the API layer."""

    return RequestContext(
        user_id=user_id,
        traits=traits or {},
        request_metadata=metadata,
        model_hint=metadata.get("model_hint"),
    )

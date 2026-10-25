from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(UTC)

from __future__ import annotations

import hashlib
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, TypeVar

T = TypeVar("T")


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_with_timeout(fn: Callable[..., T], timeout: float | None, *args: Any, **kwargs: Any) -> T:
    """Run ``fn`` in a child process so a timeout can kill it on Windows."""
    if timeout is None or timeout <= 0:
        return fn(*args, **kwargs)
    with ProcessPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout as exc:
            future.cancel()
            raise TimeoutError(f"Exceeded {timeout}s") from exc

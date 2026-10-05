from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np


@contextmanager
def measure_training_time():
    start = time.perf_counter()
    payload: dict[str, float] = {}
    try:
        yield payload
    finally:
        payload["seconds"] = time.perf_counter() - start


def measure_prediction_time(model, X, *, repeats: int = 5, per: int = 1000) -> float:
    """Median seconds per ``per`` rows after one warm-up call."""
    if len(X) == 0:
        return float("nan")
    model.predict(X[: min(len(X), 8)])
    times = []
    n = len(X)
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(X)
        elapsed = time.perf_counter() - t0
        times.append(elapsed * per / n)
    return float(np.median(times))

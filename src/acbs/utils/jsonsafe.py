"""JSON-safety helpers for API payloads.

Starlette's JSONResponse encodes with ``allow_nan=False``: a single ``NaN``
anywhere (a failed model's missing metric, an undefined ROC-AUC) turns the
whole response into a 500. Pandas ``to_dict`` produces NaN — and numpy scalar
types that ``json.dumps`` rejects outright — so payloads are sanitised on the
way out (and on the way into JSON columns, where ``NaN`` literals are illegal
on PostgreSQL jsonb): non-finite floats become ``None`` (JSON ``null``), and
numpy scalars are converted to their Python equivalents.
"""

from __future__ import annotations

import math
from typing import Any


def json_safe(obj: Any) -> Any:
    """Recursively convert ``obj`` into something JSON can represent.

    - ``NaN`` / ``±Infinity`` floats -> ``None`` (a model's missing metric
      should read as ``null``, not kill the response)
    - numpy scalars (``np.int64``, ``np.float32``, ``np.bool_``, ...) -> the
      equivalent Python scalar
    - dict / list / tuple -> same containers, sanitised element-wise
    - other unknown objects -> their ``str()`` form (never a 500)
    """
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float):  # np.float64 subclasses float
        return obj if math.isfinite(obj) else None
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if hasattr(obj, "item"):  # numpy scalars (np.int64, np.float32, ...)
        try:
            return json_safe(obj.item())
        except Exception:  # noqa: BLE001
            return _safe_str(obj)
    return _safe_str(obj)


def _safe_str(obj: Any) -> str:
    """Stringify an object without letting a hostile __str__/__repr__ raise."""
    try:
        return str(obj)
    except Exception:  # noqa: BLE001
        return "<unserializable>"

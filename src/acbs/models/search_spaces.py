from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import loguniform, randint, uniform

from acbs.utils.paths import load_yaml, repo_root


def load_search_spaces() -> dict[str, Any]:
    return load_yaml(repo_root() / "configs" / "search_spaces.yaml")


def suggest_params(trial, model_name: str) -> dict[str, Any]:
    space = load_search_spaces()[model_name]
    params: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "loguniform":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown space type {kind}")
    return params


def param_distributions(model_name: str) -> dict[str, Any]:
    space = load_search_spaces()[model_name]
    dist: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "loguniform":
            dist[name] = loguniform(spec["low"], spec["high"])
        elif kind == "float":
            dist[name] = uniform(spec["low"], spec["high"] - spec["low"])
        elif kind == "int":
            dist[name] = randint(spec["low"], spec["high"] + 1)
        elif kind == "categorical":
            dist[name] = spec["choices"]
    return dist


def sample_space(model_name: str, rng: np.random.Generator | None = None) -> dict[str, Any]:
    rng = rng or np.random.default_rng(42)
    space = load_search_spaces()[model_name]
    params: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "loguniform":
            params[name] = float(np.exp(rng.uniform(np.log(spec["low"]), np.log(spec["high"]))))
        elif kind == "float":
            params[name] = float(rng.uniform(spec["low"], spec["high"]))
        elif kind == "int":
            params[name] = int(rng.integers(spec["low"], spec["high"] + 1))
        elif kind == "categorical":
            params[name] = spec["choices"][int(rng.integers(0, len(spec["choices"])))]
    return params

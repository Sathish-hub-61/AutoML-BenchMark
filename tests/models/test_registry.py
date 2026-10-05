from __future__ import annotations

import numpy as np
import pandas as pd

from acbs.models.registry import apply_class_weight, get_models
from acbs.models.search_spaces import load_search_spaces, param_distributions, sample_space


def test_get_models_registry():
    models = get_models()
    assert len(models) >= 8
    names = [m.name for m in models]
    assert "logistic_regression" in names
    assert "xgboost" in names
    assert "random_forest" in names
    assert "svm" in names


def test_registered_models_instantiation_and_fit():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(60, 4), columns=["f1", "f2", "f3", "f4"])
    y = pd.Series([0, 1] * 30)

    for spec in get_models():
        est = spec.estimator_factory({}, 42, 1)
        apply_class_weight(est, "class_weight", y)
        est.fit(X, y)
        preds = est.predict(X)
        assert len(preds) == 60
        if spec.supports_proba and hasattr(est, "predict_proba"):
            proba = est.predict_proba(X)
            assert proba.shape == (60, 2)


def test_search_spaces_definitions():
    spaces = load_search_spaces()
    for spec in get_models():
        assert spec.name in spaces
        dists = param_distributions(spec.name)
        assert isinstance(dists, dict)
        sampled = sample_space(spec.name)
        assert isinstance(sampled, dict)

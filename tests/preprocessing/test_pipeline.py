from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

from acbs.data.profiler import profile_dataset
from acbs.preprocessing.imbalance import get_imbalance_strategy
from acbs.preprocessing.pipeline import build_full_pipeline, build_preprocessor


def test_build_preprocessor_tree_vs_linear():
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0] * 20,
            "cat": ["a", "b", "c", "a"] * 20,
        }
    )
    y = pd.Series([0, 1] * 40)
    profile = profile_dataset(df, y)

    pre_tree = build_preprocessor(profile, {}, "tree", df)
    pre_linear = build_preprocessor(profile, {}, "linear", df)

    # linear preprocessor should have scaler step in num pipeline
    num_pipe_linear = pre_linear.transformers[0][1]
    num_pipe_tree = pre_tree.transformers[0][1]

    assert "scaler" in num_pipe_linear.named_steps
    assert "scaler" not in num_pipe_tree.named_steps


def test_build_full_pipeline_smote():
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0] * 20,
            "cat": ["a", "b", "c", "a"] * 20,
        }
    )
    y = pd.Series([0] * 70 + [1] * 10)
    profile = profile_dataset(df, y)

    pre = build_preprocessor(profile, {}, "linear", df)
    pipe = build_full_pipeline(LogisticRegression(), pre, "smote")

    assert "sampler" in pipe.named_steps
    # Fit and transform
    pipe.fit(df, y)
    preds = pipe.predict(df)
    assert len(preds) == len(df)


def test_imbalance_strategy_selection():
    df = pd.DataFrame({"num": range(100)})
    y_balanced = pd.Series([0, 1] * 50)
    y_imbalanced = pd.Series([0] * 90 + [1] * 10)

    profile_bal = profile_dataset(df, y_balanced)
    profile_imb = profile_dataset(df, y_imbalanced)

    assert get_imbalance_strategy(profile_bal, {}, "auto") == "none"
    assert get_imbalance_strategy(profile_imb, {}, "auto") == "class_weight"
    assert get_imbalance_strategy(profile_imb, {"allow_smote": True}, "auto") == "smote"

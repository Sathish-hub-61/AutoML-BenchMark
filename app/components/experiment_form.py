"""Experiment form — the human uploads a CSV; the system does everything else.

Default path: pick a file → one button → the dataset is uploaded (target column
auto-detected server-side), the AutoML frameworks are chosen automatically from
installed environments, and the benchmark starts.

Everything that used to require interaction — framework checkboxes, budget,
model list, tuning, weights — lives behind the optional "Advanced settings"
expander and is entirely skippable.
"""

from __future__ import annotations

import streamlit as st

from api_client import ApiError, estimate_cost, frameworks, from_registry, registry, upload
from components.framework_picker import framework_picker

ALL_MODELS = [
    "logistic_regression",
    "svm",
    "decision_tree",
    "random_forest",
    "xgboost",
    "gradient_boosting",
    "naive_bayes",
    "knn",
]

_BUDGET_PRESETS = {
    "Auto (sized by the system)": None,
    "Quick (60 s)": 60,
    "Standard (300 s)": 300,
    "Extended (900 s)": 900,
    "Custom": -1,
}

_MAX_COST_WARNING_SECONDS = 3600  # 1 hour — warn but allow override


def _fmt_duration(s: float) -> str:
    if s < 120:
        return f"{s:.0f} seconds"
    if s < 3600:
        return f"{s / 60:.0f} minutes"
    return f"{s / 3600:.1f} hours"


def _render_profile(profile: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", profile.get("n_rows"))
    c2.metric("Features", profile.get("n_features"))
    c3.metric("Classes", profile.get("n_classes"))
    c4.metric("Imbalance", f"{profile.get('imbalance_ratio', 0):.1f}:1")
    if profile.get("warnings"):
        st.warning(" • ".join(profile["warnings"]))


def experiment_form() -> dict | None:
    """Render the new-benchmark page. Returns the request body on submit."""

    st.subheader("📂 Dataset")

    # Built-in datasets are a fallback; the primary path is uploading a CSV.
    with st.expander("…or use a built-in registry dataset"):
        try:
            names = [d["name"] for d in registry() if d.get("source") != "kaggle"]
        except Exception:
            names = ["breast_cancer", "wine"]
        choice = st.selectbox("Dataset", names, key="exp_reg_choice")
        if st.button("Load dataset", key="exp_load_btn"):
            try:
                st.session_state.dataset_payload = from_registry(choice)
                st.session_state.exp_uploaded_key = None
            except ApiError as exc:
                st.error(str(exc))

    file = st.file_uploader("CSV file", type=["csv"], key="exp_upload_file")

    # ── Advanced settings (all optional — defaults run everything automatically)
    with st.expander("⚙️ Advanced settings (optional)"):
        target_override = st.text_input(
            "Target column (leave empty to auto-detect)", key="exp_target_col"
        )
        st.subheader("Classic Models")
        models = st.multiselect(
            "Models to include", ALL_MODELS, default=ALL_MODELS, key="exp_models"
        )
        tuning = st.selectbox("Tuning strategy", ["none", "optuna", "random"], key="exp_tuning")
        cv_folds = st.slider("CV folds", 2, 5, 5, key="exp_cv_folds")
        n_trials = st.slider("Optuna trials (if tuning=optuna)", 5, 40, 10, key="exp_n_trials")

        st.subheader("🤖 AutoML Frameworks")
        manual_fw = st.checkbox(
            "Choose AutoML frameworks myself (off = system picks automatically)",
            value=False,
            key="exp_manual_fw",
        )
        selected_fws: list[str] = []
        if manual_fw:
            try:
                fw_list = frameworks()
            except Exception:
                fw_list = []
                st.warning("Could not fetch framework list from API.")
            selected_fws = framework_picker(fw_list)
        else:
            st.caption(
                "The system will select installed, OS-compatible frameworks "
                "automatically and size the budget to your dataset."
            )

        budget_label = st.radio(
            "Budget per AutoML framework",
            list(_BUDGET_PRESETS.keys()),
            horizontal=True,
            key="exp_budget_label",
        )
        preset_value = _BUDGET_PRESETS[budget_label]
        if preset_value == -1:
            budget_s: int | None = st.number_input(
                "Custom budget (seconds)", 30, 7200, 120, key="exp_custom_budget"
            )
        else:
            budget_s = preset_value

        n_jobs = st.slider("n_jobs (CPU cores for AutoML)", 1, 8, 4, key="exp_n_jobs")

        # Live cost estimate (manual mode only — in auto mode the system caps
        # the run itself, so there is nothing to confirm).
        cost_s: float | None = None
        if manual_fw and selected_fws:
            try:
                est = estimate_cost(
                    {
                        "automl_frameworks": selected_fws,
                        "automl_budget_s": budget_s or 60,
                        "models": models or ["all"],
                        "n_datasets": 1,
                    }
                )
                cost_s = est.get("total_wall_seconds", None)
            except Exception:
                pass  # API may not have /estimate yet

        st.subheader("🎯 Recommendation Preferences")
        preset = st.selectbox(
            "Priority preset",
            ["balanced", "max_accuracy", "fast_inference", "interpretable", "imbalance_safe"],
            key="exp_preset",
        )
        with st.expander("Advanced custom weights"):
            w_acc = st.slider("Accuracy", 0.0, 1.0, 0.15, key="w_acc")
            w_f1 = st.slider("F1", 0.0, 1.0, 0.25, key="w_f1")
            w_prec = st.slider("Precision", 0.0, 1.0, 0.1, key="w_prec")
            w_rec = st.slider("Recall", 0.0, 1.0, 0.15, key="w_rec")
            w_auc = st.slider("ROC-AUC", 0.0, 1.0, 0.15, key="w_auc")
            w_tr = st.slider("Train time", 0.0, 1.0, 0.1, key="w_tr")
            w_pr = st.slider("Predict time", 0.0, 1.0, 0.05, key="w_pr")
            w_int = st.slider("Interpretability", 0.0, 1.0, 0.05, key="w_int")
            use_custom = st.checkbox("Use custom weights", key="exp_use_custom")

    # ── Profile of whatever dataset is currently loaded ──────────────────────
    payload = st.session_state.get("dataset_payload")
    if payload:
        if payload.get("target"):
            src = payload.get("target_source") or "explicit"
            verb = "auto-detected" if src == "auto" else src
            st.caption(f"Target column: **{payload['target']}** ({verb})")
        if payload.get("profile"):
            _render_profile(payload["profile"])

    # ── Submit ───────────────────────────────────────────────────────────────
    cost_blocked = (
        cost_s is not None
        and cost_s > _MAX_COST_WARNING_SECONDS
        and not st.session_state.get("cost_confirmed", False)
    )
    if cost_s is not None and cost_s > _MAX_COST_WARNING_SECONDS:
        st.warning("Long run! Confirm below.")
        st.session_state.cost_confirmed = st.checkbox(
            "I understand this may take a long time", key="exp_cost_confirm"
        )

    can_run = bool(file or payload)
    button_label = "🚀 Upload & Run Benchmark" if file else "🚀 Run Benchmark"
    if not st.button(
        button_label, type="primary", disabled=(not can_run or cost_blocked), key="exp_run_btn"
    ):
        return None

    # Resolve the dataset: a fresh file always wins (upload → auto target).
    file_key = None
    if file:
        file_key = f"{file.name}:{file.size}:{target_override or ''}"
        if st.session_state.get("exp_uploaded_key") != file_key:
            try:
                payload = upload(file.name, file.getvalue(), target_override or None)
            except ApiError as exc:
                st.error(str(exc))
                return None
            st.session_state.dataset_payload = payload
            st.session_state.exp_uploaded_key = file_key
        else:
            payload = st.session_state.get("dataset_payload")
    if not payload:
        st.info("Upload a CSV or load a built-in dataset first.")
        return None

    prefs: dict = {"preset": None if use_custom else preset}
    if use_custom:
        prefs.update(
            {
                "accuracy": w_acc,
                "f1": w_f1,
                "precision": w_prec,
                "recall": w_rec,
                "roc_auc": w_auc,
                "train_time": w_tr,
                "predict_time": w_pr,
                "interpretability": w_int,
            }
        )

    body: dict = {
        "dataset_id": payload["id"],
        "config": {
            "models": models or ["all"],
            "tuning": tuning,
            "cv_folds": cv_folds,
            "n_trials": n_trials,
            "time_budget_seconds": 120,
            "seed": 42,
        },
        "preferences": prefs,
    }
    if manual_fw:
        # Explicit opt-out of automation: classic-only when nothing is picked.
        body["automl"] = {
            "frameworks": selected_fws,
            "budget_s": budget_s or 60,
            "n_jobs": n_jobs,
        }
    else:
        automl: dict = {"auto": True, "n_jobs": n_jobs}
        if budget_s:
            automl["budget_s"] = budget_s
        body["automl"] = automl
    return body

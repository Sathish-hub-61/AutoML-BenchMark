from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from api_client import (
    ApiError,
    from_registry,
    health,
    history,
    recommendation,
    registry,
    report_url,
    rerank,
    start_experiment,
    status,
    upload,
)

st.set_page_config(page_title="Auto Classifier Benchmark", layout="wide")
st.title("Auto Classifier Benchmarking System")

try:
    health()
    st.sidebar.success("API connected")
except Exception as exc:
    st.sidebar.error(f"Start the API first (`uvicorn acbs.api.main:app --port 8000`). {exc}")

page = st.sidebar.radio("Page", ["New Benchmark", "Progress", "Results", "History", "Export"])
st.sidebar.markdown(
    """
**Metrics**
- Accuracy: overall correctness (misleading when imbalanced)
- Precision / Recall / F1: class-wise quality
- ROC-AUC: ranking quality
- Train / predict time: efficiency
"""
)

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

if "experiment_id" not in st.session_state:
    st.session_state.experiment_id = None

if page == "New Benchmark":
    st.header("New Benchmark")
    source = st.radio("Data source", ["Built-in registry", "Upload CSV"], horizontal=True)
    dataset_id = None
    profile = None
    if source == "Built-in registry":
        try:
            names = [d["name"] for d in registry() if d.get("source") != "kaggle"]
        except Exception:
            names = ["breast_cancer", "wine"]
        choice = st.selectbox("Dataset", names)
        if st.button("Load dataset"):
            try:
                payload = from_registry(choice)
                st.session_state.dataset_payload = payload
            except ApiError as exc:
                st.error(str(exc))
        payload = st.session_state.get("dataset_payload")
        if payload:
            dataset_id = payload["id"]
            profile = payload["profile"]
    else:
        file = st.file_uploader("CSV", type=["csv"])
        target = st.text_input("Target column")
        if file and target and st.button("Upload"):
            path = f"data/raw/{file.name}"
            with open(path, "wb") as fh:
                fh.write(file.getbuffer())
            try:
                payload = upload(path, target)
                st.session_state.dataset_payload = payload
                dataset_id = payload["id"]
                profile = payload["profile"]
            except ApiError as exc:
                st.error(str(exc))
        payload = st.session_state.get("dataset_payload")
        if payload:
            dataset_id = payload["id"]
            profile = payload.get("profile")

    if profile:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", profile.get("n_rows"))
        c2.metric("Features", profile.get("n_features"))
        c3.metric("Classes", profile.get("n_classes"))
        c4.metric("Imbalance", f"{profile.get('imbalance_ratio', 0):.1f}:1")
        if profile.get("warnings"):
            st.warning(" • ".join(profile["warnings"]))

    models = st.multiselect("Models", ALL_MODELS, default=ALL_MODELS)
    tuning = st.selectbox("Tuning", ["none", "optuna", "random"])
    cv_folds = st.slider("CV folds", 2, 5, 5)
    n_trials = st.slider("Optuna trials", 5, 40, 10)
    time_budget = st.slider("Per-model time budget (s)", 30, 600, 120)
    preset = st.selectbox(
        "Priority preset",
        ["balanced", "max_accuracy", "fast_inference", "interpretable", "imbalance_safe"],
    )
    with st.expander("Advanced weights"):
        w_acc = st.slider("Accuracy", 0.0, 1.0, 0.15)
        w_f1 = st.slider("F1", 0.0, 1.0, 0.25)
        w_prec = st.slider("Precision", 0.0, 1.0, 0.1)
        w_rec = st.slider("Recall", 0.0, 1.0, 0.15)
        w_auc = st.slider("ROC-AUC", 0.0, 1.0, 0.15)
        w_tr = st.slider("Train time", 0.0, 1.0, 0.1)
        w_pr = st.slider("Predict time", 0.0, 1.0, 0.05)
        w_int = st.slider("Interpretability", 0.0, 1.0, 0.05)
        use_custom = st.checkbox("Use custom weights instead of preset")

    if st.button("Run benchmark", type="primary", disabled=not dataset_id):
        prefs = {"preset": None if use_custom else preset}
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
        body = {
            "dataset_id": dataset_id,
            "config": {
                "models": models or ["all"],
                "tuning": tuning,
                "cv_folds": cv_folds,
                "n_trials": n_trials,
                "time_budget_seconds": time_budget,
                "seed": 42,
            },
            "preferences": prefs,
        }
        try:
            created = start_experiment(body)
            st.session_state.experiment_id = created["id"]
            st.success(f"Experiment {created['id']} started. Open Progress.")
        except ApiError as exc:
            st.error(str(exc))

elif page == "Progress":
    st.header("Progress")
    exp_id = st.number_input("Experiment ID", min_value=1, value=int(st.session_state.experiment_id or 1))
    placeholder = st.empty()
    if st.button("Refresh"):
        st.session_state.experiment_id = int(exp_id)
    if st.session_state.experiment_id:
        try:
            s = status(int(st.session_state.experiment_id))
            prog = s.get("progress") or {}
            placeholder.progress(min(int(prog.get("pct") or 0), 100), text=prog.get("message") or s["status"])
            st.json(s)
            if s["status"] in {"queued", "running"}:
                time.sleep(2)
                st.rerun()
        except ApiError as exc:
            st.error(str(exc))

elif page == "Results":
    st.header("Results")
    exp_id = int(st.session_state.experiment_id or st.number_input("Experiment ID", min_value=1, value=1))
    try:
        rec = recommendation(exp_id)
        expn = rec.get("explanation") or {}
        st.subheader(expn.get("headline", rec.get("model_name")))
        for p in expn.get("paragraphs", []):
            st.write(p)
        st.caption(f"Stability {rec.get('stability_pct', 0):.0f}%")
        table = pd.DataFrame(rec.get("score_table") or [])
        st.dataframe(table, use_container_width=True)
        st.subheader("Live rerank (no retraining)")
        preset = st.selectbox("Preset", ["balanced", "max_accuracy", "fast_inference", "interpretable", "imbalance_safe"])
        if st.button("Apply new priorities"):
            rec = rerank(exp_id, {"preset": preset})
            st.session_state.last_rerank = rec
            st.rerun()
    except ApiError as exc:
        st.info("Run a benchmark first.")
        st.caption(str(exc))

elif page == "History":
    st.header("History")
    try:
        rows = history()
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        pick = st.number_input("Open experiment", min_value=1, value=1)
        if st.button("Open"):
            st.session_state.experiment_id = int(pick)
            st.success(f"Loaded experiment {pick}. Switch to Results.")
    except ApiError as exc:
        st.error(str(exc))

else:
    st.header("Export")
    exp_id = int(st.session_state.experiment_id or 1)
    st.markdown(f"[HTML report]({report_url(exp_id, 'html')})")
    st.markdown(f"[CSV]({report_url(exp_id, 'csv')})")
    st.markdown(f"[JSON]({report_url(exp_id, 'json')})")
    st.markdown(f"[PDF]({report_url(exp_id, 'pdf')})")

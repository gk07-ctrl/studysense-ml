"""StudySense Streamlit dashboard helpers and UI.

Helpers in this file can be imported by pytest without starting Streamlit.
The app loads saved joblib pipelines and does not retrain.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_utils import (  # noqa: E402
    CLASSIFICATION_MODEL_PATH,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    METRICS_PATH,
    REGRESSION_MODEL_PATH,
    load_model,
)

# Dashboard control ranges requested for Phase 4.
INPUT_RANGES = {
    "study_hours": (0.0, 10.0),
    "sleep_hours": (0.0, 12.0),
    "attendance": (0.0, 100.0),
    "previous_score": (0.0, 100.0),
    "assignments_completed": (0.0, 10.0),
    "concentration": (0.0, 10.0),
}

COMPARISON_CHART_PATH = FIGURES_DIR / "model_comparison.png"


def saved_model_paths():
    return {
        "regression": REGRESSION_MODEL_PATH,
        "classification": CLASSIFICATION_MODEL_PATH,
    }


def load_saved_models():
    """Load the already-trained Ridge and LogisticRegression pipelines."""
    paths = saved_model_paths()
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {role} model: {path}")
    return load_model(paths["regression"]), load_model(paths["classification"])


def load_metrics(path=None):
    metrics_path = Path(path) if path is not None else METRICS_PATH
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("metrics.json must contain a JSON object")
    return payload


def validate_dashboard_inputs(values):
    """Reject missing keys, non-numeric values, and out-of-range values."""
    missing = [name for name in FEATURE_COLUMNS if name not in values]
    if missing:
        raise ValueError(f"Missing input fields: {missing}")
    cleaned = {}
    for name in FEATURE_COLUMNS:
        try:
            number = float(values[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        low, high = INPUT_RANGES[name]
        if number < low or number > high:
            raise ValueError(f"{name} must be between {low} and {high}")
        cleaned[name] = number
    return cleaned


def prepare_feature_frame(values):
    """Return a one-row DataFrame in the same feature order used in training."""
    cleaned = validate_dashboard_inputs(values)
    return pd.DataFrame([[cleaned[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)


def predict_from_inputs(values, regression_model, classification_model):
    """Run saved pipelines. Classification is independent of the score prediction."""
    frame = prepare_feature_frame(values)
    score = float(regression_model.predict(frame)[0])
    label = int(classification_model.predict(frame)[0])
    if label not in (0, 1):
        raise ValueError(f"Classification output must be 0 or 1, got {label}")
    return score, label


def support_category(label):
    if int(label) == 1:
        return "Support recommended (demo rule)"
    return "Support not flagged (demo rule)"


def educational_recommendation(values, predicted_score, label):
    """Short study advice. Not an official academic plan."""
    cleaned = validate_dashboard_inputs(values)
    if int(label) == 1:
        return (
            f"The classifier flagged support_needed=1 for this demo input "
            f"(predicted score {predicted_score:.1f}). This is not an official "
            "assessment. As a study exercise, consider increasing regular study "
            f"time (currently {cleaned['study_hours']:.1f} hours) and checking "
            f"attendance ({cleaned['attendance']:.1f}%)."
        )
    return (
        f"The classifier flagged support_needed=0 for this demo input "
        f"(predicted score {predicted_score:.1f}). This is not an official "
        "assessment. Keep consistent study, sleep, and assignment habits; "
        "the model can still be wrong on new data."
    )


def run_streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="StudySense", layout="centered")
    st.title("StudySense")
    st.caption("Educational demo. Not an official academic assessment.")

    regression_model, classification_model = load_saved_models()
    metrics = load_metrics()

    st.header("Try a prediction")
    col_a, col_b = st.columns(2)
    with col_a:
        study_hours = st.slider("study_hours", 0.0, 10.0, 5.0, 0.1)
        sleep_hours = st.slider("sleep_hours", 0.0, 12.0, 7.0, 0.1)
        attendance = st.slider("attendance", 0.0, 100.0, 80.0, 0.5)
    with col_b:
        previous_score = st.slider("previous_score", 0.0, 100.0, 70.0, 0.5)
        assignments_completed = st.slider("assignments_completed", 0, 10, 6, 1)
        concentration = st.slider("concentration", 0.0, 10.0, 6.0, 0.1)

    values = {
        "study_hours": study_hours,
        "sleep_hours": sleep_hours,
        "attendance": attendance,
        "previous_score": previous_score,
        "assignments_completed": assignments_completed,
        "concentration": concentration,
    }

    try:
        predicted_score, label = predict_from_inputs(
            values, regression_model, classification_model
        )
        st.subheader("Result")
        st.metric("Predicted final score", f"{predicted_score:.1f}")
        st.write(f"**Support category:** {support_category(label)}")
        st.write(educational_recommendation(values, predicted_score, label))
    except (ValueError, FileNotFoundError) as exc:
        st.error(str(exc))

    st.header("Model performance")
    st.write(
        "These numbers are copied from `reports/metrics.json`. They were measured "
        "on the synthetic test split (160 train / 40 test). They are not proof of "
        "student ability."
    )
    st.write(
        f"Selected regression model: **{metrics['selected_regression_model']}**. "
        f"Selected classification model: **{metrics['selected_classification_model']}**."
    )
    st.subheader("Regression test metrics")
    st.dataframe(pd.DataFrame(metrics["regression"]).T, width="stretch")
    st.subheader("Classification test metrics")
    st.dataframe(pd.DataFrame(metrics["classification"]).T, width="stretch")
    if COMPARISON_CHART_PATH.is_file():
        st.image(
            str(COMPARISON_CHART_PATH),
            caption="Test-set model comparison saved during training (R2 and F1).",
            width="stretch",
        )
    else:
        st.info("Model comparison chart not found. Run `python src/train_models.py`.")

    st.header("About")
    st.markdown(
        """
- This app uses a **synthetic** educational dataset, not real student records.
- The support cutoff of **70** is a **demo rule**: `support_needed = 1` when
  `final_score` was below 70 in the training labels. It is not a school policy.
- Predictions here are **not official academic assessments**.
- Correlation, coefficients, and feature importance **do not prove causation**.
- The dashboard **loads saved models** from `models/` and does **not retrain**.
        """
    )


if __name__ == "__main__":
    run_streamlit_app()

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
    import streamlit.components.v1 as components

    st.set_page_config(
        page_title="StudySense",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 10% 0%, rgba(124, 58, 237, .18), transparent 30%),
                radial-gradient(circle at 90% 10%, rgba(14, 165, 233, .12), transparent 28%),
                #080b18;
        }

        .block-container {
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.03em;
        }

        .hero {
            padding: 2rem 2.2rem;
            border: 1px solid rgba(167, 139, 250, .25);
            border-radius: 28px;
            background: linear-gradient(
                135deg,
                rgba(30, 27, 75, .92),
                rgba(15, 23, 42, .78)
            );
            box-shadow: 0 20px 70px rgba(0, 0, 0, .28);
            margin-bottom: 1.5rem;
        }

        .eyebrow {
            color: #a78bfa;
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .16em;
            text-transform: uppercase;
        }

        .hero h1 {
            margin: .45rem 0 .5rem;
            font-size: clamp(2.3rem, 5vw, 4.5rem);
            line-height: .98;
            background: linear-gradient(90deg, #ffffff, #c4b5fd, #67e8f9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero p {
            color: #cbd5e1;
            font-size: 1.05rem;
            max-width: 680px;
            line-height: 1.6;
        }

        .card {
            padding: 1.2rem;
            min-height: 125px;
            border: 1px solid rgba(148, 163, 184, .16);
            border-radius: 20px;
            background: rgba(15, 23, 42, .68);
            box-shadow: 0 12px 35px rgba(0, 0, 0, .16);
        }

        .card-label {
            color: #94a3b8;
            font-size: .78rem;
            text-transform: uppercase;
            letter-spacing: .08em;
        }

        .card-value {
            color: #f8fafc;
            font-size: 1.65rem;
            font-weight: 750;
            margin-top: .45rem;
        }

        .card-note {
            color: #a5b4fc;
            font-size: .82rem;
            margin-top: .3rem;
        }

        .section-title {
            margin-top: 1.5rem;
            margin-bottom: .6rem;
            color: #e2e8f0;
            font-size: 1.25rem;
            font-weight: 700;
        }

        [data-testid="stMetric"] {
            background: rgba(15, 23, 42, .7);
            border: 1px solid rgba(167, 139, 250, .18);
            padding: 1rem;
            border-radius: 18px;
        }

        [data-testid="stSidebar"] {
            background: #0b1020;
            border-right: 1px solid rgba(148, 163, 184, .12);
        }

        div.stButton > button {
            border-radius: 12px;
            border: 1px solid rgba(167, 139, 250, .35);
            background: linear-gradient(135deg, #7c3aed, #4f46e5);
            color: white;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    regression_model, classification_model = load_saved_models()
    metrics = load_metrics()

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Personal learning intelligence</div>
            <h1>Study smarter.<br>Understand your momentum.</h1>
            <p>
                StudySense turns learning habits into a clear, responsible
                educational dashboard. Explore your study profile and receive
                practical guidance—not a final judgment.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hero_col, form_col = st.columns([1.05, 1.4], gap="large")

    with hero_col:
        components.html(
            """
            <div style="height:300px;display:grid;place-items:center;background:transparent;">
              <div class="scene">
                <div class="orb"><span></span></div>
              </div>
            </div>
            <style>
              .scene { perspective: 900px; }
              .orb {
                width: 175px;
                height: 175px;
                border-radius: 50%;
                position: relative;
                transform-style: preserve-3d;
                animation: float 4s ease-in-out infinite,
                           spin 12s linear infinite;
                background:
                  radial-gradient(circle at 28% 22%, #fff 0 3%, transparent 4%),
                  radial-gradient(circle at 35% 30%, #c4b5fd 0 8%, #7c3aed 35%, #111827 78%);
                box-shadow:
                  0 0 30px #8b5cf6,
                  0 0 95px rgba(124, 58, 237, .62);
              }

              .orb::before,
              .orb::after {
                content: "";
                position: absolute;
                inset: 13px;
                border: 2px solid rgba(255,255,255,.38);
                border-radius: 50%;
              }

              .orb::before { transform: rotateX(65deg); }
              .orb::after { transform: rotateY(65deg); }

              .orb span {
                position: absolute;
                inset: 35px;
                border-radius: 50%;
                border: 1px solid rgba(103,232,249,.7);
                transform: rotateX(65deg) rotateY(25deg);
              }

              @keyframes spin {
                from { transform: rotateY(0deg) rotateX(8deg); }
                to { transform: rotateY(360deg) rotateX(8deg); }
              }

              @keyframes float {
                0%, 100% { margin-top: 0; }
                50% { margin-top: -18px; }
              }
            </style>
            """,
            height=320,
        )

        st.markdown(
            """
            <div class="card">
                <div class="card-label">Your learning space</div>
                <div class="card-value">Focus is a skill</div>
                <div class="card-note">Small consistent actions compound.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with form_col:
        st.markdown('<div class="section-title">Build your learning profile</div>', unsafe_allow_html=True)

        with st.container(border=True):
            col_a, col_b = st.columns(2)

            with col_a:
                study_hours = st.slider("Study hours", 0.0, 10.0, 5.0, 0.1)
                sleep_hours = st.slider("Sleep hours", 0.0, 12.0, 7.0, 0.1)
                attendance = st.slider("Attendance", 0.0, 100.0, 80.0, 0.5)

            with col_b:
                previous_score = st.slider("Previous score", 0.0, 100.0, 70.0, 0.5)
                assignments_completed = st.slider("Assignments completed", 0, 10, 6, 1)
                concentration = st.slider("Concentration", 0.0, 10.0, 6.0, 0.1)

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

            st.markdown('<div class="section-title">Your current signal</div>', unsafe_allow_html=True)

            score_col, support_col = st.columns(2)

            with score_col:
                st.metric(
                    "Predicted final score",
                    f"{predicted_score:.1f}",
                    help="Educational demo estimate from the saved regression model.",
                )
                st.progress(
                    max(0.0, min(float(predicted_score) / 100.0, 1.0)),
                    text="Estimated score range",
                )

            with support_col:
                st.metric(
                    "Support signal",
                    "Recommended" if label == 1 else "Not flagged",
                    help="This is a demo classifier output, not an official assessment.",
                )
                st.caption(support_category(label))

            if label == 1:
                st.warning("Consider additional support and a consistent study routine.")
            else:
                st.success("Your current profile is not flagged by the demo rule.")

        except (ValueError, FileNotFoundError) as exc:
            st.error(str(exc))

    tabs = st.tabs(["Recommendations", "Model performance", "About"])

    with tabs[0]:
        st.markdown('<div class="section-title">A practical next step</div>', unsafe_allow_html=True)
        st.info(
            educational_recommendation(values, predicted_score, label)
        )

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-label">Study rhythm</div>
                    <div class="card-value">{study_hours:.1f} hrs</div>
                    <div class="card-note">Keep sessions consistent.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-label">Recovery</div>
                    <div class="card-value">{sleep_hours:.1f} hrs</div>
                    <div class="card-note">Protect your sleep routine.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r3:
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-label">Concentration</div>
                    <div class="card-value">{concentration:.1f}/10</div>
                    <div class="card-note">Try distraction-free blocks.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tabs[1]:
        st.caption(
            "Measured on the synthetic test split (160 train / 40 test). "
            "These metrics are not proof of student ability."
        )
        st.write(
            f"Selected regression model: **{metrics['selected_regression_model']}**"
        )
        st.write(
            f"Selected classification model: "
            f"**{metrics['selected_classification_model']}**"
        )

        st.subheader("Regression test metrics")
        st.dataframe(pd.DataFrame(metrics["regression"]).T, width="stretch")

        st.subheader("Classification test metrics")
        st.dataframe(pd.DataFrame(metrics["classification"]).T, width="stretch")

        if COMPARISON_CHART_PATH.is_file():
            st.image(
                str(COMPARISON_CHART_PATH),
                caption="Test-set model comparison saved during training.",
                width="stretch",
            )
        else:
            st.info("Model comparison chart not found.")

    with tabs[2]:
        st.markdown(
            """
            This dashboard uses a synthetic educational dataset and saved
            machine-learning pipelines. It does not retrain models.

            The support signal is a demo rule, not a school policy. Predictions
            can be wrong and should not be used as an official academic,
            medical, or admissions assessment.
            """
        )

if __name__ == "__main__":
    run_streamlit_app()

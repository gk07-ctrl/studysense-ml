"""Helpers for training, evaluating, and saving StudySense models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from data_utils import FIGURES_DIR, PROJECT_ROOT, load_student_data, validate_numeric_ranges
except ImportError:
    from src.data_utils import FIGURES_DIR, PROJECT_ROOT, load_student_data, validate_numeric_ranges

RANDOM_STATE = 42
TEST_SIZE = 0.2
SUPPORT_NEEDED_THRESHOLD = 70
N_FOREST_TREES = 50

FEATURE_COLUMNS = [
    "study_hours",
    "sleep_hours",
    "attendance",
    "previous_score",
    "assignments_completed",
    "concentration",
]
REGRESSION_TARGET = "final_score"
CLASSIFICATION_TARGET = "support_needed"

MODELS_DIR = PROJECT_ROOT / "models"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
REGRESSION_MODEL_PATH = MODELS_DIR / "best_regression.joblib"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "best_classification.joblib"


def as_project_relative(path):
    """Return a portable relative path string (no machine-specific drive letter)."""
    return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def load_validated_dataset(csv_path=None):
    """Load the CSV and run the existing range/column checks."""
    df = load_student_data(csv_path)
    validate_numeric_ranges(df)
    return df


def add_support_needed(df):
    """Flag rows that may need support: final_score below 70 -> 1, else 0."""
    out = df.copy()
    out[CLASSIFICATION_TARGET] = (out[REGRESSION_TARGET] < SUPPORT_NEEDED_THRESHOLD).astype(int)
    return out


def feature_matrix(df):
    """Return model inputs. final_score is excluded to avoid leakage."""
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[FEATURE_COLUMNS].copy()


def split_train_test(df):
    """One stratified train/test split shared by both tasks."""
    labeled = add_support_needed(df)
    X = feature_matrix(labeled)
    y_reg = labeled[REGRESSION_TARGET]
    y_clf = labeled[CLASSIFICATION_TARGET]
    return train_test_split(
        X,
        y_reg,
        y_clf,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_clf,
    )


def regression_pipelines():
    return {
        "LinearRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "DecisionTreeRegressor": Pipeline(
            [
                ("model", DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)),
            ]
        ),
    }


def classification_pipelines():
    return {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "DecisionTreeClassifier": Pipeline(
            [
                ("model", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
            ]
        ),
        "RandomForestClassifier": Pipeline(
            [
                ("model", RandomForestClassifier(
                    n_estimators=N_FOREST_TREES,
                    max_depth=5,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )),
            ]
        ),
    }


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_regression(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": _rmse(y_test, pred),
        "r2": float(r2_score(y_test, pred)),
    }


def evaluate_classification(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }


def train_regression_models(X_train, y_train, X_test, y_test):
    metrics = {}
    fitted = {}
    for name, pipe in regression_pipelines().items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        metrics[name] = evaluate_regression(pipe, X_test, y_test)
    best_name = max(metrics, key=lambda name: (metrics[name]["r2"], -metrics[name]["rmse"]))
    return fitted, metrics, best_name


def train_classification_models(X_train, y_train, X_test, y_test):
    metrics = {}
    fitted = {}
    for name, pipe in classification_pipelines().items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        metrics[name] = evaluate_classification(pipe, X_test, y_test)
    best_name = max(metrics, key=lambda name: (metrics[name]["f1"], metrics[name]["accuracy"]))
    return fitted, metrics, best_name


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Saved model not found: {path}")
    return joblib.load(path)


def write_metrics(payload, path=None):
    path = Path(path) if path is not None else METRICS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_evaluation_charts(reg_metrics, clf_metrics, best_clf, X_test, y_clf_test):
    """Save a comparison chart and a confusion matrix. No interactive windows."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    comparison_path = FIGURES_DIR / "model_comparison.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    reg_names = list(reg_metrics.keys())
    axes[0].bar(reg_names, [reg_metrics[n]["r2"] for n in reg_names], color="steelblue")
    axes[0].set_title("Regression test R2")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=20)

    clf_names = list(clf_metrics.keys())
    axes[1].bar(clf_names, [clf_metrics[n]["f1"] for n in clf_names], color="darkorange")
    axes[1].set_title("Classification test F1")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(comparison_path, dpi=100)
    plt.close(fig)

    cm_path = FIGURES_DIR / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        best_clf,
        X_test,
        y_clf_test,
        display_labels=["no support (0)", "support needed (1)"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Best classifier confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=100)
    plt.close(fig)

    return comparison_path, cm_path


def run_training():
    """Train, evaluate, save artifacts, and return the metrics payload."""
    df = load_validated_dataset()
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = split_train_test(df)

    reg_models, reg_metrics, best_reg_name = train_regression_models(
        X_train, y_reg_train, X_test, y_reg_test
    )
    clf_models, clf_metrics, best_clf_name = train_classification_models(
        X_train, y_clf_train, X_test, y_clf_test
    )

    save_model(reg_models[best_reg_name], REGRESSION_MODEL_PATH)
    save_model(clf_models[best_clf_name], CLASSIFICATION_MODEL_PATH)

    n_support = int(pd.concat([y_clf_train, y_clf_test]).sum())
    n_total = int(len(y_clf_train) + len(y_clf_test))

    payload = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_columns": FEATURE_COLUMNS,
        "regression_target": REGRESSION_TARGET,
        "classification_target": CLASSIFICATION_TARGET,
        "support_needed_threshold": SUPPORT_NEEDED_THRESHOLD,
        "support_needed_rule": (
            "support_needed is 1 when final_score is below 70, and 0 otherwise. "
            "This cutoff is a simple demo rule for this educational project, "
            "not an official academic policy."
        ),
        "class_counts": {
            "support_needed_0": n_total - n_support,
            "support_needed_1": n_support,
        },
        "regression_selection_metric": "r2",
        "classification_selection_metric": "f1",
        "selected_regression_model": best_reg_name,
        "selected_classification_model": best_clf_name,
        "metrics_path": as_project_relative(METRICS_PATH),
        "regression_model_path": as_project_relative(REGRESSION_MODEL_PATH),
        "classification_model_path": as_project_relative(CLASSIFICATION_MODEL_PATH),
        "regression": reg_metrics,
        "classification": clf_metrics,
    }
    write_metrics(payload)
    save_evaluation_charts(
        reg_metrics,
        clf_metrics,
        clf_models[best_clf_name],
        X_test,
        y_clf_test,
    )
    return payload, X_test, y_reg_test, y_clf_test

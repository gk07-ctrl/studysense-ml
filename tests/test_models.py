"""Tests for the Phase 3 training and evaluation pipeline."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model_utils import (  # noqa: E402
    CLASSIFICATION_MODEL_PATH,
    METRICS_PATH,
    REGRESSION_MODEL_PATH,
    load_model,
    run_training,
)


@pytest.fixture(scope="module")
def trained():
    payload, X_test, y_reg_test, y_clf_test = run_training()
    return {
        "payload": payload,
        "X_test": X_test,
        "y_reg_test": y_reg_test,
        "y_clf_test": y_clf_test,
    }


def test_training_produces_models(trained):
    assert REGRESSION_MODEL_PATH.is_file()
    assert CLASSIFICATION_MODEL_PATH.is_file()
    assert trained["payload"]["selected_regression_model"]
    assert trained["payload"]["selected_classification_model"]


def test_saved_models_can_be_loaded(trained):
    regression_model = load_model(REGRESSION_MODEL_PATH)
    classification_model = load_model(CLASSIFICATION_MODEL_PATH)
    assert hasattr(regression_model, "predict")
    assert hasattr(classification_model, "predict")


def test_regression_prediction_is_numeric(trained):
    model = load_model(REGRESSION_MODEL_PATH)
    preds = np.asarray(model.predict(trained["X_test"]))
    assert preds.ndim == 1
    assert np.issubdtype(preds.dtype, np.number)
    assert np.isfinite(preds).all()


def test_classification_prediction_is_0_or_1(trained):
    model = load_model(CLASSIFICATION_MODEL_PATH)
    preds = np.asarray(model.predict(trained["X_test"]))
    unique = set(preds.tolist())
    assert unique.issubset({0, 1})


def test_metrics_json_is_valid(trained):
    assert METRICS_PATH.is_file()
    payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert payload["selected_regression_model"] == trained["payload"]["selected_regression_model"]
    assert payload["selected_classification_model"] == trained["payload"]["selected_classification_model"]
    for name, values in payload["regression"].items():
        assert set(values) == {"mae", "rmse", "r2"}
        assert all(isinstance(v, float) for v in values.values())
    for name, values in payload["classification"].items():
        assert set(values) == {"accuracy", "precision", "recall", "f1"}
        assert all(isinstance(v, float) for v in values.values())

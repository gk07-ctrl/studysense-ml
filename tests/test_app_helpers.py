"""Tests for Streamlit helper functions. These do not start the dashboard."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import (  # noqa: E402
    FEATURE_COLUMNS,
    load_metrics,
    load_saved_models,
    predict_from_inputs,
    prepare_feature_frame,
    saved_model_paths,
    validate_dashboard_inputs,
)


VALID_INPUTS = {
    "study_hours": 5.0,
    "sleep_hours": 7.0,
    "attendance": 80.0,
    "previous_score": 70.0,
    "assignments_completed": 6,
    "concentration": 6.0,
}


def test_valid_input_preparation():
    frame = prepare_feature_frame(VALID_INPUTS)
    assert list(frame.columns) == FEATURE_COLUMNS
    assert len(frame) == 1
    assert frame.iloc[0]["study_hours"] == 5.0


def test_invalid_values_are_rejected():
    too_high = dict(VALID_INPUTS)
    too_high["study_hours"] = 10.1
    with pytest.raises(ValueError, match="study_hours"):
        validate_dashboard_inputs(too_high)

    too_low = dict(VALID_INPUTS)
    too_low["attendance"] = -1
    with pytest.raises(ValueError, match="attendance"):
        validate_dashboard_inputs(too_low)

    missing = dict(VALID_INPUTS)
    del missing["concentration"]
    with pytest.raises(ValueError, match="Missing input fields"):
        validate_dashboard_inputs(missing)


def test_saved_model_paths_exist():
    paths = saved_model_paths()
    assert paths["regression"].is_file()
    assert paths["classification"].is_file()


def test_prediction_output_is_numeric():
    regression_model, classification_model = load_saved_models()
    score, _label = predict_from_inputs(
        VALID_INPUTS, regression_model, classification_model
    )
    assert isinstance(score, float)
    assert np.isfinite(score)


def test_classification_output_is_0_or_1():
    regression_model, classification_model = load_saved_models()
    _score, label = predict_from_inputs(
        VALID_INPUTS, regression_model, classification_model
    )
    assert label in (0, 1)


def test_metrics_loading():
    payload = load_metrics()
    assert payload["selected_regression_model"] == "Ridge"
    assert payload["selected_classification_model"] == "LogisticRegression"
    ridge = payload["regression"]["Ridge"]
    assert set(ridge) == {"mae", "rmse", "r2"}
    assert all(isinstance(v, float) for v in ridge.values())
    raw = json.loads(Path(ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert payload["regression"]["Ridge"]["r2"] == raw["regression"]["Ridge"]["r2"]

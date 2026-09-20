"""Tests for loading and validating the student-performance dataset."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_utils import (  # noqa: E402
    DATA_PATH,
    EXPECTED_COLUMNS,
    load_student_data,
    missing_value_counts,
    validate_numeric_ranges,
    validate_required_columns,
)


def test_load_student_data_from_project_path():
    df = load_student_data()
    assert DATA_PATH.is_file()
    assert DATA_PATH == ROOT / "data" / "student_performance.csv"
    assert len(df) > 0
    assert list(df.columns) == EXPECTED_COLUMNS


def test_required_columns_present():
    df = load_student_data()
    assert validate_required_columns(df) == EXPECTED_COLUMNS


def test_required_columns_missing_raises():
    df = load_student_data().drop(columns=["final_score"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_required_columns(df)


def test_missing_values_are_zero_in_current_dataset():
    df = load_student_data()
    missing = missing_value_counts(df)
    assert int(missing.sum()) == 0


def test_missing_values_are_detected():
    df = load_student_data().copy()
    df.loc[0, "study_hours"] = None
    missing = missing_value_counts(df)
    assert int(missing["study_hours"]) == 1
    assert int(missing.sum()) == 1


def test_invalid_values_raise():
    df = load_student_data().copy()
    df.loc[0, "study_hours"] = -5
    df.loc[1, "attendance"] = 150
    with pytest.raises(ValueError, match="Numeric values outside sensible ranges"):
        validate_numeric_ranges(df)


def test_valid_ranges_pass_on_current_dataset():
    df = load_student_data()
    counts = validate_numeric_ranges(df)
    assert int(counts.sum()) == 0

"""Load and validate the StudySense student-performance dataset."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "student_performance.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

EXPECTED_COLUMNS = [
    "study_hours",
    "sleep_hours",
    "attendance",
    "previous_score",
    "assignments_completed",
    "concentration",
    "final_score",
]

# Inclusive sensible ranges for this educational synthetic dataset.
COLUMN_RANGES = {
    "study_hours": (0.0, 12.0),
    "sleep_hours": (0.0, 14.0),
    "attendance": (0.0, 100.0),
    "previous_score": (0.0, 100.0),
    "assignments_completed": (0.0, 10.0),
    "concentration": (1.0, 10.0),
    "final_score": (0.0, 100.0),
}


def load_student_data(csv_path=None):
    """Load the CSV using a project-relative path by default."""
    path = Path(csv_path) if csv_path is not None else DATA_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def validate_required_columns(df):
    """Raise ValueError if any expected column is missing."""
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return list(EXPECTED_COLUMNS)


def missing_value_counts(df):
    """Return missing-value counts for every column."""
    return df.isna().sum()


def duplicate_row_count(df):
    """Return the number of duplicate rows (excluding the first occurrence)."""
    return int(df.duplicated().sum())


def range_violation_counts(df):
    """Count values outside COLUMN_RANGES for each expected numeric column."""
    validate_required_columns(df)
    counts = {}
    for column, (low, high) in COLUMN_RANGES.items():
        series = pd.to_numeric(df[column], errors="coerce")
        invalid = series.isna() | (series < low) | (series > high)
        counts[column] = int(invalid.sum())
    return pd.Series(counts, name="range_violations")


def validate_numeric_ranges(df):
    """Raise ValueError if any numeric value is outside a sensible range."""
    counts = range_violation_counts(df)
    bad = counts[counts > 0]
    if not bad.empty:
        details = ", ".join(f"{col}={n}" for col, n in bad.items())
        raise ValueError(f"Numeric values outside sensible ranges: {details}")
    return counts


def summary_statistics(df):
    """Return pandas summary statistics for the expected columns."""
    validate_required_columns(df)
    return df[EXPECTED_COLUMNS].describe()


def validation_report(df):
    """Collect validation findings without training any model."""
    validate_required_columns(df)
    missing = missing_value_counts(df)
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_values": missing,
        "n_duplicate_rows": duplicate_row_count(df),
        "range_violations": range_violation_counts(df),
        "summary": summary_statistics(df),
    }

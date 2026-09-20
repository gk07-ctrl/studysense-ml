"""Explore and validate the student-performance dataset. No model training."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

try:
    from data_utils import (
        EXPECTED_COLUMNS,
        FIGURES_DIR,
        load_student_data,
        validate_numeric_ranges,
        validation_report,
    )
except ImportError:
    from src.data_utils import (
        EXPECTED_COLUMNS,
        FIGURES_DIR,
        load_student_data,
        validate_numeric_ranges,
        validation_report,
    )


def save_charts(df):
    """Create lightweight charts and save them under reports/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    hist_path = FIGURES_DIR / "feature_histograms.png"
    fig, axes = plt.subplots(4, 2, figsize=(10, 10))
    axes = axes.flatten()
    for i, column in enumerate(EXPECTED_COLUMNS):
        sns.histplot(df[column], bins=15, ax=axes[i], color="steelblue")
        axes[i].set_title(column)
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(hist_path, dpi=100)
    plt.close(fig)

    corr_path = FIGURES_DIR / "correlation_heatmap.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        df[EXPECTED_COLUMNS].corr(numeric_only=True),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Feature correlations (synthetic data)")
    fig.tight_layout()
    fig.savefig(corr_path, dpi=100)
    plt.close(fig)

    scatter_path = FIGURES_DIR / "study_hours_vs_final_score.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="study_hours", y="final_score", ax=ax, alpha=0.7)
    ax.set_title("Study hours vs final score (synthetic data)")
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=100)
    plt.close(fig)

    return [hist_path, corr_path, scatter_path]


def print_report(report):
    print("StudySense Phase 2: data exploration and validation")
    print(f"rows: {report['n_rows']}")
    print(f"columns: {report['n_columns']}")
    print(f"column_names: {report['columns']}")
    print("missing_values:")
    print(report["missing_values"].to_string())
    print(f"duplicate_rows: {report['n_duplicate_rows']}")
    print("range_violations:")
    print(report["range_violations"].to_string())
    print("summary_statistics:")
    print(report["summary"].to_string())


def main():
    df = load_student_data()
    report = validation_report(df)
    print_report(report)
    validate_numeric_ranges(df)
    saved = save_charts(df)
    print("saved_figures:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()

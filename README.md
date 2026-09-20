# StudySense

**Version 1.0.0**

StudySense is a lightweight student machine-learning portfolio project that predicts a synthetic exam score and a demo “support needed” label from simple study habits.

## Screenshot / demo

The dashboard is a Streamlit app. Start it with `streamlit run app.py`, then follow [docs/demo-script.md](docs/demo-script.md) for a 30–60 second walkthrough.

Training also writes charts you can show in a recording:

- `reports/figures/model_comparison.png` — test R2 and F1 by model
- `reports/figures/confusion_matrix.png` — test-set confusion matrix for the selected classifier
- `reports/figures/feature_histograms.png` — dataset distributions

## Problem statement

Students and tutors often want a small, honest example of classical ML: load a table, check data quality, train simple models, and show predictions in a UI. This project uses a **synthetic** dataset so it can run on an Intel Celeron laptop with 8 GB RAM. It is **not** an official academic assessment system.

## Features

- Load and validate a small educational CSV
- Explore missing values, duplicates, ranges, and summary statistics
- Train lightweight regression and classification models with sklearn pipelines
- Save the selected models with joblib
- Streamlit dashboard that **loads saved models** (no retraining on startup)
- Automated tests with pytest

## Project architecture

CSV data is validated in `src/data_utils.py`, explored in `src/explore_data.py`, and used by `src/train_models.py` / `src/model_utils.py` to fit pipelines and write `reports/metrics.json` plus `models/*.joblib`. `app.py` only loads those artifacts. See [docs/architecture.md](docs/architecture.md) for a diagram.

## Directory structure

```text
studysense-ml/
  app.py
  README.md
  AGENTS.md
  requirements.txt
  data/student_performance.csv
  src/data_utils.py
  src/explore_data.py
  src/model_utils.py
  src/train_models.py
  tests/test_data_utils.py
  tests/test_models.py
  tests/test_app_helpers.py
  models/best_regression.joblib
  models/best_classification.joblib
  reports/metrics.json
  reports/figures/
  docs/architecture.md
  docs/demo-script.md
```

## Dataset

`data/student_performance.csv` is a **synthetic** table (200 rows). It is **not** real student data. `final_score` was generated from a weighted combination of the other columns plus noise.

| Column | Meaning |
| --- | --- |
| `study_hours` | Hours studied per day |
| `sleep_hours` | Hours slept per night |
| `attendance` | Class attendance percent |
| `previous_score` | Prior exam score |
| `assignments_completed` | Assignments finished (out of 10) |
| `concentration` | Self-reported focus |
| `final_score` | End-of-term score (regression target) |

Classification uses a derived label `support_needed`:

- `1` when `final_score` is below **70**
- `0` otherwise

**70 is a demo cutoff only.** It is not a school policy or official definition of a student who needs support. `final_score` is never used as a model input (that would leak the label).

## Machine-learning workflow

1. Load and validate the CSV with project-relative paths.
2. Build features: the six habit columns above (no `final_score`).
3. Split 80/20 train/test with `random_state=42`, stratified on `support_needed`.
4. Fit sklearn `Pipeline` objects (StandardScaler where linear models need it).
5. Score **only on the test set**.
6. Select Ridge by highest test R2 (tie-break: lower RMSE) and LogisticRegression by highest test F1 (tie-break: accuracy).
7. Save models and metrics. The app reads those files; it does not call the trainer.

## Models tested

**Regression** (`final_score`): LinearRegression, Ridge, DecisionTreeRegressor (`max_depth=5`).

**Classification** (`support_needed`): LogisticRegression, DecisionTreeClassifier (`max_depth=5`), RandomForestClassifier (`n_estimators=50`, `max_depth=5`).

## Test metrics

Copied from `reports/metrics.json` after training with `random_state=42` (160 train / 40 test). Do not treat these as proof of student ability.

### Regression

| Model | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| LinearRegression | 3.523711273577517 | 4.493034842413609 | 0.8629607230350754 |
| Ridge | 3.514537266636512 | 4.492017928040209 | 0.8630227485678854 |
| DecisionTreeRegressor | 6.495045371295371 | 8.302600414537412 | 0.5320561334126656 |

### Classification

Class counts in the full 200-row table: `support_needed=0` → 167; `support_needed=1` → 33.

| Model | accuracy | precision | recall | F1 |
| --- | --- | --- | --- | --- |
| LogisticRegression | 0.925 | 0.8333333333333334 | 0.7142857142857143 | 0.7692307692307693 |
| DecisionTreeClassifier | 0.85 | 0.5714285714285714 | 0.5714285714285714 | 0.5714285714285714 |
| RandomForestClassifier | 0.875 | 1.0 | 0.2857142857142857 | 0.4444444444444444 |

## Why Ridge and LogisticRegression were selected

- **Ridge** had the highest test R2 (`0.8630227485678854`), slightly above LinearRegression (`0.8629607230350754`), and a slightly lower MAE and RMSE.
- **LogisticRegression** had the highest test F1 (`0.7692307692307693`). RandomForestClassifier had precision `1.0` but recall only `0.2857142857142857` on this small, imbalanced test split.

## Installation (Windows PowerShell)

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If script activation is blocked, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again. Dependencies are listed in `requirements.txt`: pandas, scikit-learn, joblib, matplotlib, seaborn, streamlit, pytest.

## Train models

```powershell
python src/explore_data.py
python src/train_models.py
```

## Run tests

```powershell
python -m pytest -v
```

## Start the Streamlit app

```powershell
streamlit run app.py
```

The app loads `models/best_regression.joblib` and `models/best_classification.joblib`. It does **not** retrain.

## Limitations and ethical considerations

- Synthetic data cannot represent a real school or university.
- Predictions are **not** official academic assessments and are **not** perfectly accurate.
- The support rule of 70 is a demo, not a policy.
- Correlation, coefficients, and feature importance **do not prove causation**.
- The positive class is uncommon (33 of 200 rows). F1, precision, and recall on 40 test rows can move a lot with a few errors.
- Linear regression can predict scores outside 0–100.
- Do not use this tool to grade, rank, or deny support to real students.

## Future improvements

- Collect a larger, consented, real (or more realistic) dataset with an ethics review
- Calibrate probabilities and report confidence intervals
- Clip or constrain regression outputs to a 0–100 scale
- Add residual plots and a simple fairness check across groups, if group labels exist
- Package a one-click Windows install script

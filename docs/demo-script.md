# StudySense demo script (30–60 seconds)

Use this while screen-recording `streamlit run app.py`. Speak in a calm, honest tone. Do not claim the tool grades real students.

## What to say (about 40 seconds)

“StudySense is a small portfolio project. It uses a synthetic student table, not real records. I trained lightweight sklearn models on a laptop: Ridge to predict a final score, and logistic regression to flag a demo label called support needed when the score was below 70. That cutoff is only a demo rule, not school policy. The dashboard loads the saved models and does not retrain. I’ll move the sliders for three profiles. Predictions can be wrong, and correlation is not causation.”

## Recording steps

1. Show the project folder briefly, then start the app (`streamlit run app.py`) and wait until **StudySense** appears.
2. Point to **About**: synthetic data, cutoff of 70, not an official assessment.
3. Point to **Model performance**: Ridge and LogisticRegression, metrics from `reports/metrics.json`.
4. Enter **Profile A**, pause on the result.
5. Enter **Profile B**, pause on the result.
6. Enter **Profile C**, pause on the result. If the score is outside 0–100, say the linear model is unconstrained.
7. Stop. Optional extra: open `reports/figures/model_comparison.png`.

## Three example input profiles

Enter these values with the sliders. Dashboard scores are shown to one decimal place, matching the app.

### Profile A — at-risk demo

| Input | Value |
| --- | --- |
| study_hours | 1.5 |
| sleep_hours | 4.5 |
| attendance | 58 |
| previous_score | 48 |
| assignments_completed | 2 |
| concentration | 3 |

**What should appear**

- Predicted final score: **46.9**
- Support category: **Support recommended (demo rule)** (`support_needed=1`)
- Recommendation text should mention this is not an official assessment and can suggest checking study time and attendance

### Profile B — balanced demo (app defaults)

| Input | Value |
| --- | --- |
| study_hours | 5.0 |
| sleep_hours | 7.0 |
| attendance | 80.0 |
| previous_score | 70.0 |
| assignments_completed | 6 |
| concentration | 6.0 |

**What should appear**

- Predicted final score: **79.8**
- Support category: **Support not flagged (demo rule)** (`support_needed=0`)
- Recommendation text should still warn that the model can be wrong

### Profile C — strong habits demo

| Input | Value |
| --- | --- |
| study_hours | 9.0 |
| sleep_hours | 8.0 |
| attendance | 95.0 |
| previous_score | 90.0 |
| assignments_completed | 10 |
| concentration | 9.0 |

**What should appear**

- Predicted final score: **111.0** (Ridge is not clipped to 0–100)
- Support category: **Support not flagged (demo rule)** (`support_needed=0`)
- Say out loud that an unconstrained linear model can leave a typical score range; that is a limitation, not a real exam mark

These three outputs were produced by loading `models/best_regression.joblib` and `models/best_classification.joblib`. If you retrain, check the numbers again before recording.

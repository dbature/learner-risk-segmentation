# Stakeholder dashboard (Module 5)

Live: https://learner-risk-segmentation.streamlit.app

A Streamlit app for programme leadership, coaching leads and the Programme
Data Ethics Committee. Six views: Overview, What drives risk, Example learners
and what-if, Learner segments, Ethical compliance, About this model.

## What it holds, and why that is safe to publish

The app reads `dashboard/data/*.json` and `models/model.joblib`. The JSON files
are built on the machine that holds the processed data:

```bash
python -m src.dashboard.build_data
```

| File | Contents | Learner rows? |
|---|---|---|
| `summary.json` | Headline metrics, capacity curve, calibration deciles, risk bands | No |
| `drivers.json` | Average SHAP effect per factor, effect by value band (cells under 10 suppressed) | No |
| `segments.json` | Module reference statistics, four centroids, segment summaries | No |
| `fairness.json` | Module 4 fairness report, and counts of test learners by score band (0.001 wide), outcome and one audited attribute at a time | No |
| `examples.json` | Learners A, B and C: day-30 features, score, segment, reasons | Three pseudonymous rows, no key, no protected attribute |

Attributes are never cross-tabulated with each other, so no small
intersection (for example an age band within a deprivation band) is exposed.
This follows the Module 2 privacy plan and the k-anonymity check in Module 3.

## How it uses the Module 4 model and API

* Scores come from the registered model file (`models/model.joblib`).
* The what-if view builds its input with the API's own `Learner` schema and
  `to_frame()` from `src/serving/api.py`, so invalid combinations are rejected
  exactly as `/predict` would reject them.
* Reasons are XGBoost TreeSHAP contributions (`src/dashboard/contrib.py`),
  tested equal to the shap library's values (`tests/test_contributions.py`).
* `scripts/api_demo.py` sends Learners A, B and C to `/predict`.

## Engagement segments

`src/segments/trajectory.py`: K-means (k = 4, fixed by the Module 1 design)
on log weekly clicks for weeks 1 to 4, standardised within module using
training-set statistics, so segments describe behaviour relative to course
peers, not course design. Names are assigned from the centroids by a fixed
rule. Silhouette is about 0.28 for k = 3, 4 and 5, so the segments are broad
patterns, reported as such.

## Deploying (Streamlit Community Cloud)

1. Sign in at https://share.streamlit.io with the GitHub account that owns the repository.
2. Create app: repository `dbature/learner-risk-segmentation`, branch `main`,
   main file `dashboard/app.py`, custom subdomain `learner-risk-segmentation`.
3. Advanced settings: Python 3.11. No secrets are needed.
4. Community Cloud installs `dashboard/requirements.txt` (it sits next to the
   entrypoint, so it takes precedence over the root `requirements.txt` and
   `environment.yml`). Every push to `main` redeploys.

The `dashboard` CI job installs the same file and renders every view
(`tests/test_dashboard_app.py`), so a broken deploy shows up in CI first.

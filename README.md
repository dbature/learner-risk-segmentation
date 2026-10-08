# Learner Risk Segmentation

Early identification and segmentation of at-risk learners so that limited instructional
coaching time reaches the learners who can still benefit from it, with fairness audited
rather than assumed.

**BAN6800 Data Analytics Capstone, Nexford University.** Author: Desmond Amos Bature.

## Problem

Instructional support is always rationed. Coaches cannot see who is disengaging until
after the outcome is recorded. This project predicts withdrawal risk from behaviour
observable early in a presentation, segments learners by engagement trajectory, and
ranks a coach caseload under a capacity constraint.

## Why fairness is the centre of this project, not a footnote

Measured by the pipeline on learners still enrolled on day 30 (27,372 registrations):

| | Most deprived IMD decile | Least deprived IMD decile |
|---|---|---|
| Still enrolled on day 30 | 79.8% | 88.2% |
| Withdrew after day 30 | 21.0% | 15.7% |
| Withdrew or failed | 55.8% | 34.6% |
| No assessment score by day 30 | 29.6% | 23.8% |

Deprived learners leave earlier, withdraw more later, and give the model less
early evidence about themselves. A model trained naively on early behaviour
learns deprivation and reports it as risk. This project measures and constrains
that disparity instead of removing the deprivation field and calling the result fair.

Earlier figures in this README (37.2% against 25.9%) were computed on every
registration, including learners who had already left by day 30. They overstated
the gap. See `docs/pipeline.md`, "Why the day-30 population exists".

## Data

Open University Learning Analytics Dataset (OULAD), CC BY 4.0.
Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Scientific Data*, 4, 170171.
Obtained from the UCI Machine Learning Repository, dataset 349.

Raw files are not committed. See `docs/data_dictionary.md` for the full schema,
measured record counts and the known data quality problems.

**This is not data from the organisation the project is written for.** OULAD is used as a
structural analogue. See the Module 1 Vision Document, Section 1.1.

## Rules this repository enforces

1. `date_unregistration`, `final_result` and `id_student` never enter the feature set.
   Checked three ways: unit test, end-to-end stage test, and the Great Expectations
   column list on the published table.
2. Train and test splits are grouped by learner. 3,538 students appear in more than
   one presentation; the pseudonymous `learner_key` keeps them grouped.
3. No raw `?` value reaches the processed layer.
4. No join may change the row count. Every join is a left join from registration.
5. Only learners still enrolled on day 30 are in the prediction population.
6. Nothing reaches `data/processed/` unless both validation suites pass.
7. Equal opportunity: recall at 25% capacity differs by at most 5 percentage points between
   groups (sex, age band, disability); for IMD band, no detectable difference (chi-square p >= 0.05).
   A model that fails this gate stays in Staging: `src/models/train.promote()` refuses Production,
   and the tests fail if a failing model is reported as passing or registered in Production.

## Data pipeline (Module 3)

Prefect flow, Great Expectations validation, DuckDB aggregation, Fairlearn
representation bias suite, HMAC pseudonymisation, privacy audit log, Docker.
Full description, lineage and run instructions: **[docs/pipeline.md](docs/pipeline.md)**.

| Document | Contents |
|---|---|
| [docs/pipeline.md](docs/pipeline.md) | Stages, lineage, how to run, scaling |
| [docs/data_dictionary.md](docs/data_dictionary.md) | Every source field and every processed field |
| [docs/governance.md](docs/governance.md) | Roles, access tiers, retention, change control |
| [docs/anonymisation_plan.md](docs/anonymisation_plan.md) | PII removal, pseudonymisation, k-anonymity |
| [reports/](reports/) | Validation, representation bias and privacy reports from the latest run |

## Predictive model (Module 4)

XGBoost predicting non-completion (withdraw or fail) for learners still enrolled on day 30,
tuned with grouped cross-validation, tracked and registered in MLflow, explained with SHAP,
LIME and DiCE, audited with Fairlearn, and served by FastAPI.

| Artefact | Location |
|---|---|
| Training, evaluation, explainability, sensitivity | `src/models/` (`python -m src.models.run_all`) |
| Pre-registered target choice | `src/models/target_check.py`, `reports/target_check.json` |
| Fairness metrics, gate and mitigation | `src/fairness/metrics.py`, `reports/fairness.json` |
| Serialized model | `models/model.joblib`, `models/model_metadata.json` |
| SHAP analysis notebook | `notebooks/05_shap_analysis.ipynb` |
| Model card | `docs/model_card.md` |
| API (`/predict`) | `src/serving/api.py` |
| Model image | `Dockerfile.model`, `requirements-model.txt` |

```bash
docker build -f Dockerfile.model -t learner-risk-model .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" -v "$PWD/reports:/app/reports" \
  -v "$PWD/mlruns:/app/mlruns" -v "$PWD/docs:/app/docs" learner-risk-model          # train and audit
docker run --rm -p 5000:5000 -v "$PWD/mlruns:/app/mlruns" learner-risk-model \
  mlflow ui --backend-store-uri sqlite:////app/mlruns/mlflow.db --host 0.0.0.0     # MLflow UI
docker run --rm -p 8000:8000 -v "$PWD/models:/app/models" learner-risk-model \
  uvicorn src.serving.api:app --host 0.0.0.0 --port 8000                           # API, docs at /docs
```

## Stakeholder dashboard and segments (Module 5)

**Live dashboard: https://learner-risk-segmentation.streamlit.app**

Streamlit app for programme leadership: headline results, what drives risk,
three example learners with a what-if panel that runs the model live, four
engagement segments (K-means, ticket #14) and an ethical compliance view that
recomputes recall by group with Fairlearn. It holds aggregates only.

| Artefact | Location |
|---|---|
| Dashboard app | `dashboard/app.py`, `dashboard/requirements.txt` |
| Dashboard data (aggregates) | `dashboard/data/`, built by `python -m src.dashboard.build_data` |
| Engagement segments | `src/segments/trajectory.py` |
| API demo | `scripts/api_demo.py` (`--in-process` runs without a server) |
| How it works, privacy, deploy steps | [docs/dashboard.md](docs/dashboard.md) |

## Quick start

```bash
pip install -r requirements-pipeline.txt
# copy every OULAD CSV, including studentVle.csv, into data/raw/
export PIPELINE_SALT="<secret, 16+ characters>"   # never commit this
python -m flows.oulad_pipeline                     # run the pipeline
pytest -q                                          # tests
```

Or in Docker:

```bash
docker build -t learner-risk-pipeline .
docker run --rm -e PIPELINE_SALT -v "$PWD/data:/app/data" -v "$PWD/logs:/app/logs" \
  -v "$PWD/reports:/app/reports" -v "$PWD/gx/uncommitted:/app/gx/uncommitted" learner-risk-pipeline
```

## Licence and attribution

Code: MIT. Data: CC BY 4.0, attribution as above.
No attempt may be made to re-identify any learner in this dataset.

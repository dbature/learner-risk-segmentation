# Learner Risk Segmentation

Early identification and segmentation of at-risk learners so that limited instructional
coaching time reaches the learners who can still benefit from it, with fairness audited
rather than assumed.

**BAN6800 Data Analytics Capstone, Nexford University.** Author: Desmond Amos Bature.

## Problem

Instructional support is always rationed. Coaches cannot see who is disengaging until
after the outcome is recorded. This project predicts non-completion (withdraw or fail)
from behaviour observable by day 30 of a presentation, segments learners by engagement
trajectory, and ranks a coach caseload under a capacity constraint. (Module 1 planned
to predict withdrawal only; why that changed is in `docs/reconciliation.md`, row 1.)

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

Streamlit app for programme leadership: headline results, a cohort view of
where risk concentrates (#20), what drives risk, three example learners with a
what-if panel that runs the model live, four engagement segments (K-means,
#14), a coach caseload that ranks an uploaded cohort under a capacity limit
(#19) and an ethical compliance view that recomputes recall by group with
Fairlearn (#21). It publishes aggregates only; the caseload demo uses 60
invented learners.

| Artefact | Location |
|---|---|
| Dashboard app | `dashboard/app.py`, `dashboard/requirements.txt` |
| Dashboard data (aggregates) | `dashboard/data/`, built by `python -m src.dashboard.build_data` |
| Engagement segments | `src/segments/trajectory.py` |
| Coach caseload | `src/dashboard/caseload.py`, demo cohort `dashboard/sample_cohort.csv` (synthetic, `scripts/make_synthetic_cohort.py`) |
| API demo | `scripts/api_demo.py` (`--in-process` runs without a server) |
| How it works, privacy, deploy steps | [docs/dashboard.md](docs/dashboard.md) |

## Final Project: the integrated system

| Component | Where |
|---|---|
| Stakeholder and explainability dashboard (SHAP, LIME, what-if, segments, fairness) | https://learner-risk-segmentation.streamlit.app |
| Monitoring dashboard (performance, Evidently drift, fairness over time, response plan) | https://learner-risk-monitoring.streamlit.app |
| Prediction API on Render (`/predict`, `/explanations/{id}`, docs at `/docs`; free tier, so the first request after 15 idle minutes takes about a minute) | https://learner-risk-api.onrender.com |
| CI/CD | `.github/workflows/ci.yml`: lint and tests, three image builds, dashboard render tests, Compose smoke test, then deploy of the API to Render from `main` only |
| Multi-container setup | `docker-compose.yml` (below) |
| Monitoring build | `src/monitoring/build.py` writes `monitoring/data/monitoring.json` |
| Gated retraining with MLflow | `src/models/retrain.py` writes `reports/retraining_decision.json` |
| Explanation store (audit trail of every prediction) | `src/governance/explanation_store.py` |
| Cross-module contradictions and how each was resolved | [docs/reconciliation.md](docs/reconciliation.md) |
| Implementation and monitoring plan, KPIs | [docs/monitoring.md](docs/monitoring.md) |
| Model card and limitations | [docs/model_card.md](docs/model_card.md) |
| Screenshots (monitoring, XAI view, MLflow retraining run, architecture) | [docs/evidence/final/](docs/evidence/final/) |

Ethics and governance documents, in [docs/ethics/](docs/ethics/):
[ethical AI framework](docs/ethics/ethical_ai_framework.md),
[algorithmic impact assessment](docs/ethics/algorithmic_impact_assessment.md),
[regulatory compliance](docs/ethics/regulatory_compliance.md),
[public trust statement](docs/ethics/public_trust_statement.md),
[governance structure](docs/ethics/governance_structure.md),
[incident response](docs/ethics/incident_response.md),
[decommissioning](docs/ethics/decommissioning.md) and the
[release review pack](docs/ethics/release_review_pack.md) for ticket #23.

**Status.** The model is registered in Staging. It fails two fairness checks
(disability recall gap 5.1 points after mitigation; recall differs across IMD
bands, p = 0.008), and the Ethics Committee sign-off (#23) is pending. What is
public is a demonstrator on open data, not an operational release.

```bash
docker compose up -d                       # API :8000, dashboard :8501, monitoring :8502, MLflow :5000
cp .env.example .env                       # then set PIPELINE_SALT; never commit .env
docker compose --profile batch run --rm pipeline          # rebuild data/processed
docker compose --profile batch run --rm monitoring-build  # refresh monitoring.json
docker compose --profile batch run --rm retrain           # train and assess a challenger
```

Continuous deployment needs a Render web service built from
`deploy/api/Dockerfile` (auto-deploy off) and its deploy hook URL stored as the
repository secret `RENDER_DEPLOY_HOOK`. The deploy job runs only after every
check passes on `main`; without the secret it is skipped. The API was first
planned for a Hugging Face Docker Space, but those now need a paid PRO plan;
`scripts/deploy_hf_space.py` is kept for anyone who has one.

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

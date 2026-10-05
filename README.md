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
7. Recall parity across IMD bands stays within 5 percentage points, or the build fails
   (Module 4).

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

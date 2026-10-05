# Data Pipeline: Stages, Lineage and How to Run It

Module 3 deliverable. The pipeline turns the seven OULAD source tables into a
validated, pseudonymised, model-ready table for Module 4. It is orchestrated by
Prefect (`flows/oulad_pipeline.py`), validated by Great Expectations (`gx/`),
containerised with Docker (`Dockerfile`), and every stage is unit tested
(`tests/`).

## Stage graph

```
ingest ────────┐
               ├─> clean ─> integrate ─> anonymise ─> validate (GX) ─> promote
aggregate_vle ─┘                                            └─> bias_check ─> privacy_report
```

| # | Stage | Module | What it does | Fails the run when |
|---|---|---|---|---|
| 1 | ingest | `src/data/ingest.py` | Reads six tables with latin-1 encoding; only `?` becomes null; types pinned; Parquet out | a file's columns differ from the expected schema |
| 2 | aggregate_vle | `src/features/early_window.py` | DuckDB streams 10.6M click rows into one row per registration, days before 30 only | the source file is missing |
| 3 | clean | `src/data/clean.py` | Fixes the `10-20` IMD label, orders categoricals, checks duplicates, quarantines 102 label conflicts | a duplicated key carries different content; an unknown category appears |
| 4 | integrate | `src/data/integrate.py` | Left joins from registration, builds features and both labels, defines the day-30 population | any join changes the row count |
| 5 | anonymise | `src/governance/anonymise.py` | HMAC pseudonym replaces `id_student`; splits sensitive attributes into a restricted file; writes to staging | the salt is missing; a PII-like column appears |
| 6 | validate | `src/data/validate.py` | Two GX suites, 92 expectations | any expectation fails |
| 7 | promote | `src/governance/anonymise.py` | Moves staged files into `data/processed` | never runs unless validation passed |
| 8 | bias_check | `src/fairness/representation.py` | Representation bias by IMD, disability, gender, age, education | configurable (`bias.fail_on_flag`) |
| 9 | privacy_report | `src/governance/anonymise.py` | k-anonymity on quasi-identifiers | n/a, reported |

Every stage writes to the privacy audit log (`logs/privacy_audit.jsonl`).

## Lineage, measured on the October 2026 run

| Layer | Dataset | Rows | Notes |
|---|---|---|---|
| raw | studentInfo, studentRegistration | 32,593 each | 1,111 `?` in imd_band; 45 in date_registration; 22,521 in date_unregistration (means "never unregistered", not missing) |
| raw | studentAssessment | 173,912 | 173 `?` scores |
| raw | studentVle | 10,655,280 | 433 MB, aggregated out of core |
| interim | vle_early_window | 28,815 | registrations with VLE activity before day 30 |
| quarantine | label_conflicts | 102 | 93 Withdrawn with no unregistration date; 9 Fail with one |
| interim | integrated | 32,593 | one row per registration, 35 columns, row count asserted |
| processed | model_ready | 27,372 | day-30 population, 19 features, 2 labels, pseudonymous key |
| processed | audit_attributes | 27,372 | sensitive attributes only, fairness auditor access |
| processed | early_leavers | 5,127 | left on or before day 30; 2,678 before the course began |

27,372 = 32,593 registrations, less 5,127 who left by day 30, less 94 label
conflicts inside the remaining population.

## Why the day-30 population exists

Half of all withdrawals (5,127 of 10,072) happen on or before day 30, and the
median withdrawal day is 27. On the cleaned table, learners who eventually
withdrew were 9.1 times more likely than the rest to show no VLE activity in
days 0 to 29. Among learners still enrolled on day 30, the ratio is 1.16. The
difference was learners who had already left being scored as "at risk". The
population rule removes that leak before any model sees the data (Kaufman et al., 2012).

## Run the pipeline

### Locally

```bash
pip install -r requirements-pipeline.txt
# copy all OULAD CSVs, including studentVle.csv, into data/raw/
export PIPELINE_SALT="<a secret of 16+ characters>"      # PowerShell: $env:PIPELINE_SALT="..."
python -m flows.oulad_pipeline
```

### In Docker

```powershell
docker build -t learner-risk-pipeline .
docker run --rm `
  -e PIPELINE_SALT=$env:PIPELINE_SALT -e PIPELINE_ACTOR=desmond `
  -v "${PWD}/data:/app/data" -v "${PWD}/logs:/app/logs" `
  -v "${PWD}/reports:/app/reports" -v "${PWD}/gx/uncommitted:/app/gx/uncommitted" `
  learner-risk-pipeline

docker run --rm learner-risk-pipeline pytest -q      # tests inside the image
```

Validation results open in `gx/uncommitted/data_docs/local_site/index.html`.

## Scaling and collaboration

* **Volume.** The only large table is aggregated by DuckDB without loading it
  into memory. Swapping the CSV path for Parquet or a warehouse table changes
  one `read_csv` call.
* **New sources.** A new table needs a schema entry in `ingest.py`, a left join
  in `integrate.py`, and expectations in `validate.py`. The row-count assertion
  and the GX column list catch a join or column that was forgotten.
* **Scheduling.** The same flow can be deployed to a Prefect server with a cron
  schedule; nothing in the code assumes a manual run.
* **Collaboration.** Thresholds live in `params.yaml`, so changing one is a
  reviewed commit. CI runs lint and the full test suite on every push and builds the
  Docker image. DVC (`dvc.yaml`) versions each stage's outputs.

## References

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data
mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge
Discovery from Data, 6*(4), Article 15. https://doi.org/10.1145/2382577.2382579

Polyzotis, N., Zinkevich, M., Roy, S., Breck, E., & Whang, S. (2019). Data
validation for machine learning. *Proceedings of Machine Learning and Systems, 1*,
334-347.

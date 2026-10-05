# Project Board: sprints and tickets

Four two-week sprints, mapped to the remaining capstone modules.
Create these as issues on a GitHub Project (board view) with columns
**Backlog / In progress / In review / Done** and one iteration field per sprint.

Labels used: `data`, `model`, `fairness`, `infra`, `docs`, `gate`.

---

## Sprint 1: Foundation (Module 2 close out)
| # | Ticket | Labels | Definition of done |
|---|---|---|---|
| 1 | Initialise repository with the Module 1 directory structure | infra | Tree matches Section 4.2 of the Vision Document |
| 2 | Pin environment (`requirements.txt`, `environment.yml`) | infra | Clean install reproduces on a second machine |
| 3 | Write data dictionary from measured profiling | docs, data | Every field, record count and `?` count verified against the files |
| 4 | Draft Data Protection Impact Assessment | docs, gate | Tabled at Sprint 1 review for Ethics Committee approval |
| 5 | CI skeleton with ruff and pytest | infra | Build runs green on push |

## Sprint 2: Data pipeline (Module 3)
| # | Ticket | Labels | Definition of done |
|---|---|---|---|
| 6 | `ingest.py`: latin-1 encoding pinned, `?` converted to null | data | Regression test covers the encoding failure |
| 7 | Great Expectations suite: schema, ranges, eleven IMD levels | data | Pipeline fails if a raw `?` reaches `processed/` |
| 8 | Normalise `imd_band`, including the `10-20` label defect | data | Ordinal sort returns deciles in order |
| 9 | DuckDB aggregation of `studentVle` to a 30 day window | data | 10.6M rows reduced to one row per student |
| 10 | `build_features.py`: first assessment, score, registration timing | data, model | Features documented in the data dictionary |
| 11 | Leakage gate: exclude `date_unregistration`, split by `id_student` | gate | `tests/test_features.py` fails if either rule is broken |

## Sprint 3: Model and audit (Module 4)
| # | Ticket | Labels | Definition of done |
|---|---|---|---|
| 12 | Baseline: logistic regression on first assessment submission alone | model | Benchmark recorded in MLflow |
| 13 | XGBoost with probability calibration | model | Beats baseline on recall at a fixed selection rate |
| 14 | K-means segmentation on engagement trajectory | model | Segments are interpretable and differ in recommended action |
| 15 | SHAP explanations per learner | model, fairness | Every flag carries a readable reason |
| 16 | Fairlearn group metrics by IMD band and disability | fairness | Recall, selection rate and FPR reported per group |
| 17 | Recall parity constraint within 5 percentage points | fairness, gate | `tests/test_fairness.py` blocks release above tolerance |
| 18 | Model card recording subgroup performance and limits | docs, gate | Follows Mitchell et al. (2019) |

## Sprint 4: Serving and release (Module 5 and Final)
| # | Ticket | Labels | Definition of done |
|---|---|---|---|
| 19 | Streamlit coach caseload, ranked under a capacity constraint | infra | No more than 25% of a cohort flagged |
| 20 | Cohort view showing where risk concentrates | infra | Withdrawal rate and mean risk score render by module presentation, IMD band and prior education, with cell counts shown so small groups are not read as signal |
| 21 | User facing fairness panel | fairness | Visible to coaches, not buried in documentation |
| 22 | Deploy to Streamlit Community Cloud | infra | Public URL live |
| 23 | Ethics Committee release review | gate | Sign off recorded before public deployment |

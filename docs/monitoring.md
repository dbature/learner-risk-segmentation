# Implementation and monitoring plan

Live monitoring dashboard: https://learner-risk-monitoring.streamlit.app
(built by `python -m src.monitoring.build`, app in `monitoring/app.py`).

## 1. Deployment path

| Phase | What runs | Gate to the next phase |
|---|---|---|
| 0. Demonstrator (now) | Public dashboard, monitoring dashboard and `/predict` API on open data; model in Staging | Ethics Committee release review (ticket #23), DPIA approved |
| 1. Rebuild on programme data | Pipeline and model rebuilt on the programme's own records; same tests, gates and model card | Hold-out AUC at least 0.70, every fairness gate passed, DPIA re-approved |
| 2. Pilot | 3 to 5 coaches, one term; caseload behind a login with role-based views; weekly caseload, flags not carried forward | Coaches act on at least 60% of flags within a week; no high fairness alert open |
| 3. Scale | All coaches in the area; monthly monitoring review | Quarterly committee review continues |

Infrastructure is already containerised (`docker-compose.yml`): pipeline,
trainer, API, dashboard, monitoring and MLflow run on one machine; the public
copies run on Streamlit Community Cloud and a Hugging Face Space, deployed by
GitHub Actions after every check passes.

## 2. KPIs

| KPI | Target | Source | Cadence |
|---|---|---|---|
| ROC AUC on completed cohort | at least 0.70 (alert below) | monitoring build | each course run |
| Non-completers reached at capacity | within 5 points of validation | monitoring build | each course run |
| Flags that were right (precision) | at least 70% | monitoring build | each course run |
| Share flagged | 15% to 35% (planned 25%) | API and monitoring | weekly |
| Recall gap by sex, age, disability | at most 5 points on the running total | monitoring build | each course run |
| IMD equal-recall test | p at least 0.05 on the running total | monitoring build | each course run |
| Score drift (PSI) | below 0.10 watch, 0.25 alert | monitoring build | each course run |
| Pipeline validation pass rate | 100% | Great Expectations | every pipeline run |
| Flags actioned within a week (pilot) | at least 60% | coach log | weekly |
| Learners with a readable reason | 100% | explanation store | continuous |

## 3. Continuous fairness monitoring

Fairness is recomputed with Fairlearn for every batch under both the single
threshold and the mitigated thresholds. Single batches are too small to
judge (groups of 50 to 300 non-completers), so a batch gap above 5 points is
a **watch** item unless a chi-square test shows it is real (p < 0.05), in
which case it is a **high** alert. The release gate is applied to the
**running total** of all monitored learners. Today the running total repeats
the validation finding: disability gap 5.1 points and IMD p = 0.008, both
high alerts, which is why the model is in Staging.

## 4. Model drift and bias drift

* **Prediction drift:** population stability index of the risk score against
  the training distribution (all batches 0.01 to 0.04, stable).
* **Data drift:** Evidently's DataDriftPreset on all 20 inputs. 5 to 14 inputs
  drift per batch, mostly because each course run has a different course
  mix, so input drift alone never triggers retraining; it triggers a look.
* **Bias drift:** the running-total recall gaps and IMD test over time
  (monitoring dashboard, Fairness view). A gap that widens across three
  consecutive runs is escalated even before it crosses 5 points.

## 5. Retraining policy

Retraining is a decision triggered by an alert, not a schedule.
`src/models/retrain.py` retrains with the champion's registered settings on
the refreshed data, logs the challenger to MLflow and registers it in
Staging only if it loses no more than 0.01 AUC or 2 points of recall, and it
is eligible for release review only if every fairness gate also passes.
Production is never set by the script.

Trial on unchanged data (`reports/retraining_decision.json`): in the Linux
container the challenger reproduced the champion exactly; on the Windows
laptop it came out marginally different (AUC 0.7596 against 0.7596, recall
44.7% against 44.8%, threshold 0.591 against 0.593), so retraining is
deterministic within one platform, not across platforms. That run's
challenger passed the disability gate but still failed IMD, so it was
registered as a challenger and marked not eligible. The champion's disability gap
of 5.1 points crossed back under the 5-point line when the thresholds moved
by about 0.002, which shows how close to the line it sits (release review
question 2).

## 6. Explanation store

Every `/predict` call is written to an append-only JSON-lines store
(`src/governance/explanation_store.py`) with the model hash and version, the
request hash and inputs, the score, threshold, rule and every feature's
contribution. An auditor retrieves a record by id (`GET /explanations/{id}`);
nothing lists other callers' records. In production the store sits on
access-controlled storage and is kept for the retention period in
`docs/governance.md`.

## 7. Responsibilities

Measurement analyst: runs the monitoring build each course run and reviews
alerts within 5 working days. Measurement lead: owns retraining decisions.
Ethics Committee: quarterly review of the monitoring dashboard; any high
fairness alert goes to the committee (see `docs/ethics/incident_response.md`).

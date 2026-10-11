# Ethical AI Framework

One document that ties every ethical commitment made in Modules 1 to 5 to the
control that enforces it and the evidence a reviewer can check. It follows the
four functions of the NIST AI Risk Management Framework (Govern, Map,
Measure, Manage; NIST, 2023).

## Principles (Module 1 Ethical AI Charter)

1. **Fairness.** The system must not concentrate its errors on the learners it
   exists to serve. Disparity is measured and reported at every release.
2. **Transparency.** No learner is flagged without a reason a coach can read
   and challenge. Model card and fairness report are public.
3. **Accountability.** A named human makes every intervention decision. The
   model recommends; it never allocates.
4. **Proportionality.** Collect only what the decision needs; drop any feature
   that adds accuracy without actionable insight.

## Principles to controls to evidence

| Principle | Control (code or process) | Evidence |
|---|---|---|
| Fairness | Protected attributes never model inputs; equal-opportunity gate (5 points; chi-square for IMD) that blocks promotion; capacity-constrained mitigation | `tests/test_fairness.py`, `src/models/train.py::promote`, `reports/fairness.json` |
| Fairness | Representation bias suite in the pipeline | `reports/representation_bias.json` |
| Fairness | Continuous fairness monitoring on the running total | `monitoring/data/monitoring.json`, monitoring dashboard |
| Transparency | SHAP reasons on every score (dashboard, API); LIME cross-check; counterfactuals | Dashboard, `/predict` `top_factors`, `notebooks/05_shap_analysis.ipynb` |
| Transparency | Model card generated from the run's own reports | `docs/model_card.md` |
| Transparency | Public trust statement | `docs/ethics/public_trust_statement.md` |
| Accountability | Model in Staging until the Ethics Committee rules; no script sets Production | Registry, `docs/ethics/release_review_pack.md` |
| Accountability | Append-only explanation store for every prediction | `src/governance/explanation_store.py` |
| Accountability | Incident response and decommissioning plans | `docs/ethics/incident_response.md`, `decommissioning.md` |
| Proportionality | 20 day-30 features; no names, no learner keys in the API | `src/models/data.py`, API schema |
| Privacy | HMAC pseudonymisation, k-anonymity check, privacy audit log, tiered access | `src/governance/`, `reports/privacy_report.json`, `docs/governance.md` |

## Govern, Map, Measure, Manage

* **Govern:** Programme Data Ethics Committee with release authority
  (`governance_structure.md`); change control in `docs/governance.md`.
* **Map:** algorithmic impact assessment (`algorithmic_impact_assessment.md`):
  who is affected, how, and what could go wrong.
* **Measure:** validation, fairness gate, monitoring KPIs (`docs/monitoring.md`).
* **Manage:** alerts mapped to responses, incident response, gated retraining,
  decommissioning.

## Where the framework currently blocks release

The fairness gate fails on disability (5.1 points against a 5-point
tolerance) and on IMD band (p = 0.008), and the DPIA was not approved
before modelling (`docs/reconciliation.md`, item 5). The framework did its
job: the model is registered, explained and monitored, and it is not
released.

## Reference

National Institute of Standards and Technology. (2023). *Artificial intelligence
risk management framework (AI RMF 1.0)* (NIST AI 100-1).
https://doi.org/10.6028/NIST.AI.100-1

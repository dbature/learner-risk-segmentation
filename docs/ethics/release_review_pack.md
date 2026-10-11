# Release review pack (ticket #23): pending

**Status: pending. No sign-off has been recorded.** The model
(learner-risk-noncompletion v1, alias champion, Staging) is not released for
use with real learners. The public dashboard and API are demonstrators on
open data.

## Evidence for the committee

* Model card: `docs/model_card.md`
* Fairness report: `reports/fairness.json`, Module 4 Fairness Metrics Report
* Monitoring: https://learner-risk-monitoring.streamlit.app
* Impact assessment: `docs/ethics/algorithmic_impact_assessment.md`
* Reconciliation log: `docs/reconciliation.md`

## Questions the committee must decide

1. **IMD.** Recall is highest in the most deprived bands (51% against 38%).
   Is recall that leans towards the learners the programme exists to reach
   acceptable, or must IMD-aware thresholds be added? Should the chi-square
   rule for many-level attributes be ratified?
2. **Disability.** A 5.1-point gap in favour of disabled learners, not
   statistically detectable (p = 0.13). A real breach of the 5-point rule,
   or within tolerance?
3. **Sex at the point of decision.** The mitigation sets separate score lines
   for women and men. Accept this differential treatment, or use the
   in-processing alternative that costs 0.059 AUC?

## Conditions any approval must carry

DPIA approved; model rebuilt and validated on programme data; pilot limited
to 3 to 5 coaches with role-based access; monthly monitoring review; this
pack re-submitted for every new model version.

## Decision record

| Date | Decision | Votes | Conditions | Minute reference |
|---|---|---|---|---|
| | *Pending* | | | |

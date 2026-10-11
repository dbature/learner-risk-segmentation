# Incident response plan for ethical failures

## What counts as an incident

| Severity | Examples |
|---|---|
| **High** | Statistically detectable recall gap above 5 points on the running total; a learner harmed by a flag (stigma, exclusion); a flag used outside purpose; personal data exposed; scores visible to learners |
| **Medium** | Batch-level fairness watch item for three consecutive runs; AUC below 0.70 on a completed cohort; flag rate outside 15% to 35% |
| **Low** | Single-batch fairness watch item; input drift without a performance change |

## Response

1. **Detect.** Monitoring alert, coach report, or data subject complaint.
2. **Contain (high: within 1 working day).** Suspend new flags: the API can be
   switched off and the caseload hidden without touching the data. Existing
   flags are reviewed by a coach, not deleted.
3. **Notify.** Measurement lead within 1 working day; committee chair within
   5; the DPO immediately if personal data is involved (NDPA breach
   notification duties apply in deployment).
4. **Investigate.** Use the explanation store to retrieve the exact model
   version, inputs and reasons for every affected prediction; reproduce with
   the tagged release.
5. **Remediate.** Options in order of preference: fix data, adjust threshold
   (committee approval), retrain via `src/models/retrain.py`, or decommission.
6. **Review.** Written note to the committee within 20 working days; the
   reconciliation log and model card are updated.

## Current open incident

None from use, because the model is not in use. The two high fairness
alerts (disability 5.1 points; IMD p = 0.008) were found at validation and
are being handled through the release review rather than the incident path.

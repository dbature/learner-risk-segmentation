# Decommissioning plan

**When.** Any of: AUC below 0.70 on two consecutive completed cohorts after
retraining; a high fairness incident that remediation cannot fix; the
committee withdraws approval; the programme ends; the data regime changes so
that processing is no longer lawful.

**How.**

1. Committee decision recorded; coaches told one week ahead, with the reason.
2. Stop new scoring: disable the API deployment and hide the caseload view;
   the monitoring page states the retirement date.
3. Hand over: coaches return to their previous allocation method with the
   last cohort view, so no learner loses support abruptly.
4. Archive: model file, model card, fairness report, monitoring history and
   registry entry kept with the release record, marked Archived in MLflow.
5. Delete: processed data and the explanation store after the retention
   period in `docs/governance.md`; rotate and destroy the pseudonymisation
   salt so archived pseudonyms can no longer be linked.
6. Close-out note to the committee: what the model did, what was learned,
   and whether a successor is justified.

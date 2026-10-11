# Regulatory compliance report

The brief asks for GDPR, CCPA and HIPAA. The source data is UK learner data
and the intended deployment is Nigeria, so the UK GDPR and the Nigeria Data
Protection Act 2023 are the regimes that would actually bite. Each is
assessed below as applies, partly applies or does not apply.

| Regime | Applies? | Why | What the project does |
|---|---|---|---|
| GDPR / UK GDPR | **Yes** for OULAD-like data on identifiable learners | Pseudonymised data is still personal data (Recital 26) | Lawful basis and purpose limitation (Art. 5, 6); special category data (disability) processed only for equality monitoring (Art. 9; UK DPA 2018 Sch. 1 para. 8); data minimisation and protection by design (Art. 5(1)(c), 25); DPIA (Art. 35); transparency about the logic used (Art. 13 to 15). Art. 22 (solely automated decisions) is **not triggered** because a coach makes every decision, and that is a design commitment, not an accident |
| Nigeria Data Protection Act 2023 and GAID 2025 | **Yes** in the deployment context | Programme operates in Nigeria | DPIA before high-risk processing including automated decision-making; Data Protection Officer designated; annual Compliance Audit Return by 31 March (Nigeria Data Protection Commission, 2025); same minimisation and retention rules (`docs/governance.md`) |
| CCPA / CPRA (California) | **No** | Applies to for-profit businesses meeting revenue or data-volume thresholds that collect California residents' personal information; the programme is neither | Practices aligned anyway: notice of purpose, no sale or sharing of data, deletion on request |
| HIPAA | **No** | Covers health plans, providers, clearinghouses and their business associates; education records are excluded from protected health information (45 C.F.R. § 160.103) | Disability is still treated as sensitive: separate storage tier, audit use only |
| Equality Act 2010 (UK) | Context | Sex, age and disability are protected characteristics (s. 4) | Audited, never used as inputs; sex used only by the mitigation step, flagged as differential treatment for committee decision |

## Data subject rights (deployment)

Right to be informed (privacy notice to learners and caregivers), access
(learners can obtain their risk record and reasons from the explanation
store by id), rectification (corrected source data flows through the next
pipeline run), objection (a learner can be excluded from scoring), and
erasure at the end of retention. None of these is exercised in the capstone
because no real programme learner is scored.

## Open items

1. DPIA approval (late; see reconciliation item 5).
2. DPO designation and privacy notice for the deployment organisation.
3. Data processing agreement with any cloud host before real data is used;
   the public demonstrators hold no personal data.

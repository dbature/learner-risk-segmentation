# Algorithmic Impact Assessment (including DPIA content)

Completes ticket #4 late: the DPIA should have been approved before
modelling and was not (`docs/reconciliation.md`, item 5). It is written for
the deployment case, where the data would be personal data about real
programme learners.

## 1. System and purpose

Ranks learners still enrolled on day 30 of a course by risk of not
completing, with reasons, so coaches can prioritise limited contact time.
Decision support only: a named coach decides; the system never contacts a
learner, removes support or affects grades.

## 2. Data

Day-30 behaviour (assessments due and submitted, early marks, weekly online
activity) and registration facts (course, credits, previous attempts,
registration timing). Sex, age band, disability and deprivation band are
held separately and used only to audit fairness. Disability is special
category data (GDPR Art. 9). Learner identifiers are pseudonymised with a
keyed hash; names never enter any layer.

## 3. Who is affected and how

| Group | Possible benefit | Possible harm |
|---|---|---|
| Learners flagged correctly | Support while the outcome can still change | Being labelled; lowered expectations |
| Learners flagged wrongly (22 in 100 flags) | Extra contact they did not need | Same labelling risk |
| Learners missed (about 55 in 100 non-completers) | None from the tool | False reassurance if coaches stop using judgement |
| Most deprived learners | Reached more often (51% against 38%) | Over-attention could feel like surveillance |
| Disabled learners | Reached slightly more often after mitigation | Disability is used in auditing, so a breach would be sensitive |
| Coaches | Time spent where it matters | Automation bias; workload if capacity is set wrong |

## 4. Risk assessment

| Risk | Likelihood | Impact | Mitigation | Residual |
|---|---|---|---|---|
| Unequal reach by group | Medium (measured) | High | Gate, mitigation, monitoring | **High: gate fails on disability and IMD** |
| Labelling and stigma | Medium | High | Support framing; no learner-facing score; weekly regeneration | Medium |
| Automation bias (missed learners ignored) | Medium | High | Coach overrides; Learner C shown in training material; "a low score is not a guarantee" in the UI | Medium |
| Re-identification | Low | High | Pseudonymisation, k-anonymity, aggregates-only public outputs | Low |
| Drift to a new population | High | High | Revalidation on programme data before use; monitoring | Medium |
| Use outside purpose (e.g. admissions) | Low | High | Model card out-of-scope list; committee approval for any new use | Low |

## 5. Necessity and proportionality

Reaching 45% of non-completers at 25% capacity, against 25% by chance, is a
material gain for a scarce resource. The 20 inputs are those a coach could
observe anyway; no feature is collected for the model alone.

## 6. Decision

Not approved for operational use. Conditions for approval are in
`release_review_pack.md`.

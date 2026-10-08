# Model Card: Learner Non-Completion Risk at Day 30

Registered as `learner-risk-noncompletion` version 1, alias `champion`, stage **Staging**. Generated automatically from `reports/*.json` by `src/models/model_card.py`.

## Model details
* Algorithm: XGBoost (Chen and Guestrin, 2016) inside a scikit-learn pipeline (median imputation with missing-value indicators, scaling, one-hot encoding of module).
* Tuned hyperparameters: subsample = 0.85, reg_lambda = 1, n_estimators = 200, min_child_weight = 10, max_depth = 4, learning_rate = 0.05, colsample_bytree = 0.6.
* Developed by Desmond Amos Bature, BAN6800 Data Analytics Capstone, Nexford University, October 2026.
* Training data: Open University Learning Analytics Dataset (Kuzilek et al., 2017), processed by the Module 3 pipeline. Learner identifiers are pseudonymised.

## Intended use
* Rank learners still enrolled on day 30 of a course by risk of not completing (withdrawing or failing), so that a coach with capacity for about a quarter of the cohort reaches the learners most likely to need help.
* Decision support for a named human. The model never contacts a learner and never removes support.
* Out of scope: admissions, grading, any sanction, and learners who left before day 30 (they are a separate registration-time segment).

## Factors
Fairness is audited across sex, age band, disability and IMD (deprivation) band. None of these is a model input. Sex is used only by the post-processing step that equalises recall.

## Metrics
Ranking quality (ROC AUC, average precision) and the operating point a coach uses: flag the riskiest 25%, then measure recall (share of non-completers reached) and precision (share of flags that are right).

## Evaluation data
Grouped 80/20 hold-out: 5,455 registrations, no learner on both sides. Non-completion rate 43.4%. Time-aware check: trained on 2013B to 2014B, tested on 7,849 new learners in 2014J.

## Quantitative analysis
| Model | ROC AUC | Avg precision | Recall at 25% | Precision at 25% |
|---|---|---|---|---|
| Majority class | 0.500 | 0.434 | 0.0% | n/a |
| Random flagging | 0.503 | 0.438 | 24.9% | 43.2% |
| Registration data only | 0.592 | 0.521 | 31.3% | 54.3% |
| First assessment only | 0.601 | 0.593 | 33.7% | 58.6% |
| Logistic regression | 0.743 | 0.713 | 41.6% | 73.9% |
| Random forest | 0.757 | 0.737 | 45.3% | 78.6% |
| XGBoost (selected) | 0.760 | 0.739 | 44.8% | 78.1% |

Time-aware ROC AUC 0.710; seed stability 0.759, 0.760, 0.760; with 25% input noise on click counts, 1.1% of flags change.

### Fairness (test set, recall at 25% capacity)
| Attribute | Recall gap before | Recall gap after | Disparate impact ratio | Chi-square p | Gate rule | Gate |
|---|---|---|---|---|---|---|
| Sex (recorded as gender) | 0.121 | 0.013 | 0.94 | 0.55 | recall gap <= 0.05 | pass |
| Age band | 0.036 | 0.042 | 0.82 | 0.07 | recall gap <= 0.05 | pass |
| Disability | 0.003 | 0.051 | 0.80 | 0.13 | recall gap <= 0.05 | FAIL |
| IMD band | 0.131 | 0.139 | 0.61 | 0.01 | chi-square p >= 0.05 (many-level) | FAIL |

Groups with fewer than 100 positive cases in the test set are reported but not gated (age 55 and over; missing IMD).

IMD band: the literal Module 1 rule (largest minus smallest recall, 0.139) cannot test ten bands. A perfectly fair model would show a median gap of 0.101 at this sample size and pass it only 1.3% of the time. The gate for many-level attributes is therefore a chi-square test of equal recall, pending Ethics Committee ratification. IMD fails it (p = 0.008): recall is highest in the most deprived bands (51.4% in 0-10%) and lowest in the least deprived (38.1% in 90-100%).

## Release decision
The fairness gate fails on Disability, IMD band. The model stays in **Staging** and `src/models/train.promote()` refuses Production until the gate passes or the Ethics Committee formally changes the rule.

## Ethical considerations
* The model learns from behaviour that is itself socially patterned: deprived learners are flagged more often because they do not complete more often. Recall parity, not equal flag rates, is the commitment.
* Mitigation uses sex at the point of decision. That is differential treatment and needs Ethics Committee approval before release. The in-processing alternative that avoids it costs 0.06 AUC.
* A flag is an invitation to support, never a judgement. Coaches see reasons (SHAP), not just scores.

## Caveats and recommendations
* Recall at 25% capacity is 44.8% before mitigation and 44.7% after, short of the Module 2 target of 70%. The target was set before learners who had already left were removed from the population (Module 3). Acceptance criteria are reset against the baselines above.
* Top drivers: Mean score, assessments by day 30, Share of early assessments submitted, Active days, days 0 to 29, Clicks, week 4, Module BBB. Early assessment behaviour dominates, so modules with nothing due by day 30 get weaker predictions.
* The data is UK adult distance learning used as an analogue. Revalidate before any use on another population.
* Monitor monthly: score drift, recall by group on completed cohorts, and the share of learners with no early assessment.

# Data Dictionary

Source: Open University Learning Analytics Dataset (OULAD), CC BY 4.0.
Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Scientific Data*, 4, 170171.
Obtained from the UCI Machine Learning Repository, dataset 349.

All row counts, value counts and missingness figures below were measured directly
from the downloaded files, not taken from the published description.
Files are **latin-1 encoded** and fail to parse as UTF-8.
Missing values are encoded as the literal string `?`, not as blanks or nulls.

---

## courses.csv: 22 rows, 3 columns
| Field | Type | Notes |
|---|---|---|
| code_module | categorical (7) | Module code, AAA to GGG |
| code_presentation | categorical (4) | 2013B, 2013J, 2014B, 2014J. B = February start, J = October start |
| module_presentation_length | integer (7 distinct) | Length of presentation in days, 234 to 269 |

## assessments.csv: 206 rows, 6 columns
| Field | Type | Notes |
|---|---|---|
| code_module | categorical (7) | Join key |
| code_presentation | categorical (4) | Join key |
| id_assessment | integer (206 distinct) | Primary key |
| assessment_type | categorical (3) | TMA (tutor marked), CMA (computer marked), Exam |
| date | mixed | Days from presentation start. **11 rows hold `?`** |
| weight | float (24 distinct) | Percentage contribution. Exams weight 100 |

## vle.csv: 6,364 rows, 6 columns
| Field | Type | Notes |
|---|---|---|
| id_site | integer (6,364 distinct) | Primary key for a learning resource |
| code_module, code_presentation | categorical | Join keys |
| activity_type | categorical (20) | resource, oucontent, url, forumng, quiz and others |
| week_from | mixed | **5,243 of 6,364 rows hold `?` (82.4%)**, so this field is unusable without imputation |
| week_to | mixed | Same, 5,243 rows hold `?` |

## studentInfo.csv: 32,593 rows, 12 columns
| Field | Type | Notes |
|---|---|---|
| code_module, code_presentation | categorical | Join keys |
| id_student | integer | **28,785 distinct students across 32,593 rows.** 3,538 students appear in more than one presentation, so splits must be made by student |
| gender | categorical (2) | M 17,875, F 14,718 |
| region | categorical (13) | UK and Ireland regions |
| highest_education | categorical (5) | No Formal quals to Post Graduate Qualification |
| imd_band | categorical (11) | Index of Multiple Deprivation decile. **1,111 rows hold `?`**. One level is labelled `10-20` without a percent sign, inconsistent with the other ten. **Protected attribute for fairness analysis** |
| age_band | categorical (3) | 0-35, 35-55, 55<= |
| num_of_prev_attempts | integer (7 distinct) | Prior attempts at the module |
| studied_credits | integer (61 distinct) | Credits enrolled |
| disability | categorical (2) | N 29,429, Y 3,164. **Protected attribute for fairness analysis** |
| final_result | categorical (4) | **Target source.** Pass 12,361, Withdrawn 10,156, Fail 7,052, Distinction 3,024. **Disagrees with date_unregistration on 102 rows** (93 Withdrawn with no date, 9 Fail with one); these are quarantined |

## studentRegistration.csv: 32,593 rows, 5 columns
| Field | Type | Notes |
|---|---|---|
| code_module, code_presentation, id_student | | Join keys |
| date_registration | mixed | Days relative to presentation start, negative means before. **45 rows hold `?`** |
| date_unregistration | mixed | **22,521 rows hold `?`; 10,072 carry a value (30.9%) against a withdrawal rate of 31.2%. This field encodes the target. It defines the label and must never enter the feature set** |

## studentAssessment.csv: 173,912 rows, 5 columns
| Field | Type | Notes |
|---|---|---|
| id_assessment | integer (188 distinct) | 18 of the 206 assessments have no submissions |
| id_student | integer | **23,369 distinct students. 5,416 of the 28,785 students never submitted a single assessment** |
| date_submitted | integer | Days from presentation start |
| is_banked | binary | Score transferred from a previous presentation |
| score | mixed | 0 to 100. **173 rows hold `?`** |

## studentVle.csv: 10,655,280 rows, 6 columns
Not committed and not transferred: the file is 433 MB.
Aggregated out of core by DuckDB (`src/features/early_window.py`) to one row per registration
covering days before 30. No other stage reads it.

| Field | Type | Notes |
|---|---|---|
| code_module, code_presentation, id_student, id_site | | Join keys |
| date | integer | Day of interaction relative to presentation start |
| sum_click | integer | Clicks by that student on that resource on that day |

---

## Processed layer: `data/processed/model_ready.parquet` (Module 3)

27,372 rows, one per registration still enrolled on day 30, 24 columns. Produced by
`flows/oulad_pipeline.py` and validated by the `model_ready` Great Expectations suite,
whose column list must match this table exactly.

| Field | Type | Definition |
|---|---|---|
| learner_key | string (16 hex) | HMAC-SHA256 pseudonym of id_student. Same learner, same key. Group splits on this |
| code_module, code_presentation | categorical | Context for grouping and time-based splits. Not features |
| num_of_prev_attempts | integer | From studentInfo |
| studied_credits | integer | From studentInfo, 30 to 655 |
| date_registration | integer, nullable | Days relative to start; 7 missing in the population |
| date_registration_missing | 0/1 | Flag for the above |
| n_due_by_30 | integer | Non-exam assessments with a deadline on or before day 30. **0 for 4,980 learners**, whose module sets nothing that early |
| n_submitted_by_30 | integer | Of those, submitted by day 30 (banked counts as submitted) |
| n_banked_by_30 | integer | Of those, banked from an earlier attempt |
| submit_rate_by_30 | float, nullable | n_submitted / n_due; null when nothing was due |
| mean_score_by_30 | float, nullable | Mean score of those submissions; null for 7,195 learners |
| clicks_pre_start | integer | VLE clicks before day 0 |
| clicks_0_29 | integer | VLE clicks, days 0 to 29 |
| clicks_wk1 to clicks_wk4 | integer | Days 0-6, 7-13, 14-20, 21-29 |
| active_days_0_29 | integer | Distinct days with any click, 0 to 30 |
| distinct_sites_0_29 | integer | Distinct resources touched |
| first_active_day | integer | First day with a click, 0 to 29; **-1 means never active**, not day 0 |
| no_vle_activity | 0/1 | No VLE activity at all before day 30 (861 learners) |
| label_withdrew_after_30 | 0/1 | Unregistered after day 30. Rate 18.1% |
| label_non_completion | 0/1 | final_result Withdrawn or Fail. Rate 43.8% |

Module 4 chooses between the two labels on evidence. Neither label, nor
date_unregistration, final_result or id_student, may appear among the features;
three independent checks enforce this (unit test, stage test, GX column list).

## Other processed and quarantine files

| File | Rows | Contents | Access tier |
|---|---|---|---|
| `audit_attributes.parquet` | 27,372 | learner_key, context, gender, disability, imd_band (`10-20` fixed to `10-20%`), age_band, region, highest_education | 3, fairness auditor |
| `early_leavers.parquet` | 5,127 | Learners who left on or before day 30, with registration-time fields and the day they left | 2 |
| `data/quarantine/label_conflicts.parquet` | 102 | Registrations whose outcome and unregistration date contradict each other, with the reason | 1, steward |

# Data Dictionary

Source: Open University Learning Analytics Dataset (OULAD), CC BY 4.0.
Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Scientific Data*, 4, 170171.
Obtained from the UCI Machine Learning Repository, dataset 349.

All row counts, value counts and missingness figures below were measured directly
from the downloaded files, not taken from the published description.
Files are **latin-1 encoded** and fail to parse as UTF-8.
Missing values are encoded as the literal string `?`, not as blanks or nulls.

---

## courses.csv — 22 rows, 3 columns
| Field | Type | Notes |
|---|---|---|
| code_module | categorical (7) | Module code, AAA to GGG |
| code_presentation | categorical (4) | 2013B, 2013J, 2014B, 2014J. B = February start, J = October start |
| module_presentation_length | integer (7 distinct) | Length of presentation in days, 234 to 269 |

## assessments.csv — 206 rows, 6 columns
| Field | Type | Notes |
|---|---|---|
| code_module | categorical (7) | Join key |
| code_presentation | categorical (4) | Join key |
| id_assessment | integer (206 distinct) | Primary key |
| assessment_type | categorical (3) | TMA (tutor marked), CMA (computer marked), Exam |
| date | mixed | Days from presentation start. **11 rows hold `?`** |
| weight | float (24 distinct) | Percentage contribution. Exams weight 100 |

## vle.csv — 6,364 rows, 6 columns
| Field | Type | Notes |
|---|---|---|
| id_site | integer (6,364 distinct) | Primary key for a learning resource |
| code_module, code_presentation | categorical | Join keys |
| activity_type | categorical (20) | resource, oucontent, url, forumng, quiz and others |
| week_from | mixed | **5,243 of 6,364 rows hold `?` (82.4%)**, so this field is unusable without imputation |
| week_to | mixed | Same, 5,243 rows hold `?` |

## studentInfo.csv — 32,593 rows, 12 columns
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
| final_result | categorical (4) | **Target source.** Pass 12,361, Withdrawn 10,156, Fail 7,052, Distinction 3,024 |

## studentRegistration.csv — 32,593 rows, 5 columns
| Field | Type | Notes |
|---|---|---|
| code_module, code_presentation, id_student | | Join keys |
| date_registration | mixed | Days relative to presentation start, negative means before. **45 rows hold `?`** |
| date_unregistration | mixed | **22,521 rows hold `?`; 10,072 carry a value (30.9%) against a withdrawal rate of 31.2%. This field encodes the target. It defines the label and must never enter the feature set** |

## studentAssessment.csv — 173,912 rows, 5 columns
| Field | Type | Notes |
|---|---|---|
| id_assessment | integer (188 distinct) | 18 of the 206 assessments have no submissions |
| id_student | integer | **23,369 distinct students. 5,416 of the 28,785 students never submitted a single assessment** |
| date_submitted | integer | Days from presentation start |
| is_banked | binary | Score transferred from a previous presentation |
| score | mixed | 0 to 100. **173 rows hold `?`** |

## studentVle.csv — 10,655,280 rows, 5 columns
Not committed and not transferred: the file is 433 MB.
Aggregated locally to a per student early window summary before use.

| Field | Type | Notes |
|---|---|---|
| code_module, code_presentation, id_student, id_site | | Join keys |
| date | integer | Day of interaction relative to presentation start |
| sum_click | integer | Clicks by that student on that resource on that day |

---

## Derived fields

| Field | Definition | Purpose |
|---|---|---|
| `withdrew` | `final_result == "Withdrawn"` | Binary target |
| `submitted_first_assessment` | Student has a row in studentAssessment for the earliest non exam assessment of their presentation | Earliest strong predictor. Non submitters withdraw at 77.8%, submitters at 17.9% |
| `first_assessment_score` | Score on that assessment | Distinction 84.2, Pass 75.7, Fail 67.2, Withdrawn 64.9 |
| `clicks_30`, `active_days_30` | Clicks and distinct active days in the first 30 days, from studentVle | Early engagement, built locally |
| `days_before_start` | `date_registration`, negative values | Withdrawers register earlier, median -67 against -53 for Pass |

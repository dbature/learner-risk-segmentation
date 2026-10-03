# Learner Risk Segmentation

Early identification and segmentation of at-risk learners so that limited instructional
coaching time reaches the learners who can still benefit from it, with fairness audited
rather than assumed.

**BAN6800 Data Analytics Capstone, Nexford University.** Author: Desmond Amos Bature.

## Problem

Instructional support is always rationed. Coaches cannot see who is disengaging until
after the outcome is recorded. This project predicts withdrawal risk from behaviour
observable early in a presentation, segments learners by engagement trajectory, and
ranks a coach caseload under a capacity constraint.

## Why fairness is the centre of this project, not a footnote

Measured directly from the data:

| | Most deprived IMD decile | Least deprived IMD decile |
|---|---|---|
| Withdrawal rate | 37.2% | 25.9% |
| Distinction rate | 5.1% | 14.1% |
| Submits first assessment | 71.2% | 83.2% |

The early signal is itself socially patterned. A model trained naively on engagement
learns deprivation and reports it as risk. This project measures and constrains that
disparity instead of removing the deprivation field and calling the result fair.

## Data

Open University Learning Analytics Dataset (OULAD), CC BY 4.0.
Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Scientific Data*, 4, 170171.
Obtained from the UCI Machine Learning Repository, dataset 349.

Raw files are not committed. See `docs/data_dictionary.md` for the full schema,
measured record counts and the known data quality problems.

**This is not data from the organisation the project is written for.** OULAD is used as a
structural analogue. See the Module 1 Vision Document, Section 1.1.

## Rules this repository enforces in CI

1. `date_unregistration` never enters the feature set. It encodes the target.
2. Train and test splits are made by `id_student`, never by row. 3,538 students appear
   in more than one presentation.
3. No raw `?` value reaches the processed layer.
4. Recall parity across IMD bands stays within 5 percentage points, or the build fails.

## Quick start

```bash
conda env create -f environment.yml && conda activate learner-risk
# or: pip install -r requirements.txt

dvc repro            # rebuild the pipeline
pytest -q            # tests, including the leakage and fairness gates
streamlit run src/dashboard/app.py
```

## Licence and attribution

Code: MIT. Data: CC BY 4.0, attribution as above.
No attempt may be made to re-identify any learner in this dataset.

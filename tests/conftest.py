"""Fixtures for the data rule gates.

These use a small frame that mirrors the real OULAD schema rather than the
dataset itself, so the gates are enforceable in CI from the first commit,
before the pipeline in Modules 3 onwards exists. The rules under test are
fixed; only the data source changes later.
"""
import pandas as pd
import pytest

# Columns the feature build is allowed to produce. date_unregistration,
# final_result and withdrew are deliberately absent: see tests/test_features.py.
ALLOWED_FEATURES = [
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
    "date_registration",
    "submitted_first_assessment",
    "first_assessment_score",
    "clicks_30",
    "active_days_30",
]


@pytest.fixture
def feature_columns():
    return ALLOWED_FEATURES


@pytest.fixture
def student_frame():
    """Five learners, two of whom appear in two presentations each, mirroring
    the 3,538 students who appear more than once in the real data."""
    return pd.DataFrame(
        {
            "id_student": [11391, 11391, 28400, 30268, 30268, 31604, 32885],
            "code_presentation": ["2013J", "2014J", "2013J", "2013B", "2014B", "2013J", "2014J"],
            "imd_band": ["90-100%", "90-100%", "20-30%", "0-10%", "0-10%", "50-60%", "30-40%"],
            "withdrew": [0, 0, 1, 1, 0, 0, 1],
        }
    )


@pytest.fixture
def grouped_split(student_frame):
    """A split made on id_student, which is the rule under test."""
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    train_idx, test_idx = next(
        splitter.split(student_frame, groups=student_frame["id_student"])
    )
    return (
        student_frame.iloc[train_idx]["id_student"].tolist(),
        student_frame.iloc[test_idx]["id_student"].tolist(),
    )


@pytest.fixture
def train_ids(grouped_split):
    return grouped_split[0]


@pytest.fixture
def test_ids(grouped_split):
    return grouped_split[1]


@pytest.fixture
def processed_frame():
    """Processed output must never carry the raw '?' sentinel. Six source
    fields use it instead of nulls."""
    return pd.DataFrame(
        {
            "imd_band": ["90-100%", "20-30%", None],
            "date_registration": [-53.0, -67.0, None],
            "first_assessment_score": [78.0, None, 64.0],
        }
    )

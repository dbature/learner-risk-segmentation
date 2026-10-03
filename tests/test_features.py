"""Leakage gates. These are the tests that fail the build, not style checks."""
import pandas as pd
import pytest

LEAKY = {"date_unregistration", "final_result", "withdrew"}


def test_unregistration_never_a_feature(feature_columns):
    """date_unregistration is present in 30.9% of rows against a 31.2% withdrawal
    rate. It encodes the target and must only be used to construct the label."""
    assert not (LEAKY & set(feature_columns)), f"leaky columns present: {LEAKY & set(feature_columns)}"


def test_split_is_by_student(train_ids, test_ids):
    """3,538 of 28,785 students appear in more than one presentation. A row level
    split trains and tests on the same person."""
    assert not (set(train_ids) & set(test_ids)), "student overlap between train and test"


def test_no_raw_sentinel_in_processed(processed_frame):
    """Missing values are encoded as the literal string '?' in six source fields."""
    offenders = [c for c in processed_frame.columns
                 if processed_frame[c].astype(str).str.strip().eq("?").any()]
    assert not offenders, f"raw '?' reached the processed layer in: {offenders}"

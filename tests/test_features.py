"""Leakage gates. These fail the build, and they are not style checks."""

LEAKY = {"date_unregistration", "final_result", "withdrew"}


def test_unregistration_never_a_feature(feature_columns):
    """date_unregistration is present in 30.9% of rows against a 31.2%
    withdrawal rate. It encodes the target and may only build the label."""
    leaked = LEAKY & set(feature_columns)
    assert not leaked, f"leaky columns present in the feature set: {sorted(leaked)}"


def test_split_is_by_student(train_ids, test_ids):
    """3,538 of 28,785 students appear in more than one presentation, so a
    row level split would train and test on the same person."""
    overlap = set(train_ids) & set(test_ids)
    assert not overlap, f"student overlap between train and test: {sorted(overlap)}"


def test_no_raw_sentinel_in_processed(processed_frame):
    """Missing values arrive as the literal string '?' in six source fields
    and must be converted before anything downstream sees them."""
    offenders = [
        column
        for column in processed_frame.columns
        if processed_frame[column].astype(str).str.strip().eq("?").any()
    ]
    assert not offenders, f"raw '?' reached the processed layer in: {offenders}"

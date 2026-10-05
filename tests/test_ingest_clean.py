"""Ingest and clean rules, each traced to a problem found in the real files."""
import pandas as pd
import pytest

from src.data.clean import (
    DataQualityError,
    check_duplicates,
    label_conflicts,
    normalise_imd_band,
)
from src.data.ingest import SchemaError, read_oulad_csv


def test_latin1_file_reads_and_question_mark_becomes_null(tmp_path):
    """Regression for ticket 6: the files are latin-1 and break a UTF-8 reader."""
    path = tmp_path / "studentRegistration.csv"
    path.write_bytes(
        "code_module,code_presentation,id_student,date_registration,date_unregistration\n"
        "AAA,2013J,1,-50,?\n".encode("latin-1")
    )
    df = read_oulad_csv(path, "studentRegistration")
    assert df["date_unregistration"].isna().all()
    assert str(df["date_unregistration"].dtype) == "Int64"


def test_non_ascii_byte_does_not_break_ingest(tmp_path):
    path = tmp_path / "courses.csv"
    path.write_bytes("code_module,code_presentation,module_presentation_length\n"
                     "\xc9AA,2013J,268\n".encode("latin-1"))
    with pytest.raises(UnicodeDecodeError):
        pd.read_csv(path, encoding="utf-8")
    assert read_oulad_csv(path, "courses").loc[0, "code_module"] == "\xc9AA"


def test_only_question_mark_is_missing(tmp_path):
    """'NA' is pandas' default null marker; here it must survive as text."""
    path = tmp_path / "vle.csv"
    path.write_bytes("id_site,code_module,code_presentation,activity_type,week_from,week_to\n"
                     "1,AAA,2013J,NA,?,?\n".encode("latin-1"))
    df = read_oulad_csv(path, "vle")
    assert df.loc[0, "activity_type"] == "NA"
    assert df["week_from"].isna().all()


def test_unexpected_columns_fail_loudly(tmp_path):
    path = tmp_path / "courses.csv"
    path.write_text("code_module,presentation,length\nAAA,2013J,268\n", encoding="latin-1")
    with pytest.raises(SchemaError):
        read_oulad_csv(path, "courses")


def test_imd_label_defect_is_normalised_and_ordered():
    out = pd.Series(normalise_imd_band(pd.Series(["10-20", "0-10%", None])))
    assert out.tolist()[:2] == ["10-20%", "0-10%"]
    assert pd.isna(out.iloc[2])
    assert out.cat.ordered and out.min() == "0-10%"


def test_unknown_imd_label_is_rejected():
    with pytest.raises(DataQualityError):
        normalise_imd_band(pd.Series(["100-110%"]))


def test_conflicting_duplicate_keys_are_not_dropped_silently():
    df = pd.DataFrame({"id_site": [1, 1], "code_module": ["AAA", "BBB"],
                       "code_presentation": ["2013J", "2013J"], "activity_type": ["a", "a"],
                       "week_from": [None, None], "week_to": [None, None]})
    with pytest.raises(DataQualityError):
        check_duplicates(df, "vle")


def test_label_conflicts_found_in_both_directions():
    key = {"code_module": ["AAA"] * 3, "code_presentation": ["2013J"] * 3, "id_student": [1, 2, 3]}
    info = pd.DataFrame({**key, "final_result": ["Withdrawn", "Fail", "Pass"]})
    reg = pd.DataFrame({**key, "date_unregistration": pd.array([None, 0, None], dtype="Int64")})
    out = label_conflicts(info, reg)
    assert set(out["reason"]) == {"withdrawn_without_unregistration_date",
                                  "unregistration_date_but_not_withdrawn"}
    assert sorted(out["id_student"]) == [1, 2]

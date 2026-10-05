"""Fixtures for the data rule gates.

These use small frames that mirror the real OULAD schema rather than the
dataset itself, so every gate runs in CI without the licensed data. The
tiny_raw fixture writes a complete miniature raw layer, in the same latin-1
encoding and with the same '?' sentinel, so the pipeline stages can be run
end to end in a temporary folder.
"""
import pandas as pd
import pytest


@pytest.fixture
def feature_columns():
    """The real feature list from the pipeline, not a copy of it, so the
    leakage gate tests what the model will actually see."""
    from src.features.build_features import FEATURE_COLUMNS

    return FEATURE_COLUMNS


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


@pytest.fixture(autouse=True)
def _test_salt(monkeypatch):
    monkeypatch.setenv("PIPELINE_SALT", "test-salt-not-a-secret-0123456789")
    monkeypatch.setenv("PIPELINE_ACTOR", "pytest")


def _write(path, text):
    path.write_bytes(text.encode("latin-1"))


@pytest.fixture
def tiny_raw(tmp_path):
    """A miniature OULAD raw layer covering every rule the pipeline enforces.

    Learners, all on AAA 2013J:
      1  active, submits the day-20 TMA, passes
      2  unregisters on day 10 (early leaver, outside the population)
      3  unregisters on day 60 (label_withdrew_after_30 = 1), no VLE before day 30
      4  final_result Withdrawn but no unregistration date (quarantined)
      5  imd_band "?" and date_registration "?", clicks only on day 40
    """
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    _write(raw / "courses.csv",
           "code_module,code_presentation,module_presentation_length\nAAA,2013J,268\n")
    _write(raw / "assessments.csv",
           "code_module,code_presentation,id_assessment,assessment_type,date,weight\n"
           "AAA,2013J,1,TMA,20,10.0\n"
           "AAA,2013J,2,TMA,60,20.0\n"
           "AAA,2013J,3,Exam,?,100.0\n")
    _write(raw / "studentInfo.csv",
           "code_module,code_presentation,id_student,gender,region,highest_education,"
           "imd_band,age_band,num_of_prev_attempts,studied_credits,disability,final_result\n"
           "AAA,2013J,1,M,East Anglian Region,HE Qualification,90-100%,55<=,0,240,N,Pass\n"
           "AAA,2013J,2,F,Scotland,A Level or Equivalent,10-20,35-55,0,60,N,Withdrawn\n"
           "AAA,2013J,3,F,Wales,Lower Than A Level,0-10%,0-35,1,60,Y,Withdrawn\n"
           "AAA,2013J,4,M,London Region,Lower Than A Level,20-30%,0-35,0,120,N,Withdrawn\n"
           "AAA,2013J,5,M,Ireland,Lower Than A Level,?,0-35,0,60,N,Fail\n")
    _write(raw / "studentRegistration.csv",
           "code_module,code_presentation,id_student,date_registration,date_unregistration\n"
           "AAA,2013J,1,-50,?\n"
           "AAA,2013J,2,-20,10\n"
           "AAA,2013J,3,-30,60\n"
           "AAA,2013J,4,-10,?\n"
           "AAA,2013J,5,?,?\n")
    _write(raw / "studentAssessment.csv",
           "id_assessment,id_student,date_submitted,is_banked,score\n"
           "1,1,18,0,78\n"
           "1,3,35,0,60\n"
           "2,1,58,0,81\n")
    _write(raw / "vle.csv",
           "id_site,code_module,code_presentation,activity_type,week_from,week_to\n"
           "100,AAA,2013J,resource,?,?\n101,AAA,2013J,forumng,?,?\n")
    _write(raw / "studentVle.csv",
           "code_module,code_presentation,id_student,id_site,date,sum_click\n"
           "AAA,2013J,1,100,-5,3\n"
           "AAA,2013J,1,100,0,4\n"
           "AAA,2013J,1,101,2,6\n"
           "AAA,2013J,1,100,25,2\n"
           "AAA,2013J,2,100,1,1\n"
           "AAA,2013J,4,101,3,5\n"
           "AAA,2013J,5,100,40,9\n")
    return tmp_path

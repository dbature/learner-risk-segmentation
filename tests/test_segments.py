"""Segment model: naming rule, persistence, and a privacy check on what it stores."""
import numpy as np
import pandas as pd
import pytest

from src.segments.trajectory import ORDER, WEEKS, TrajectorySegmenter


def _learners(n=400, seed=0):
    rng = np.random.default_rng(seed)
    shapes = {"steady": [90, 90, 95, 95], "slipping": [60, 20, 10, 5], "late": [0, 10, 60, 40], "gone": [0, 0, 0, 0]}
    rows = []
    for i in range(n):
        base = list(shapes.values())[i % 4]
        rows.append([max(0, int(b * rng.uniform(0.7, 1.3))) for b in base] + [["AAA", "BBB"][(i // 4) % 2]])
    return pd.DataFrame(rows, columns=WEEKS + ["code_module"])


def test_names_follow_the_centroid_rule():
    X = _learners()
    seg = TrajectorySegmenter().fit(X, sample=400)
    assert sorted(seg.names.values()) == sorted(ORDER)
    got = seg.predict(pd.DataFrame([[95, 95, 100, 100, "AAA"], [0, 0, 0, 0, "BBB"], [70, 15, 5, 2, "AAA"],
                                    [0, 15, 70, 45, "BBB"]], columns=WEEKS + ["code_module"]))
    assert list(got) == ["Steady", "Disengaged", "Slipping", "Late starter"]


def test_round_trip_gives_identical_assignments():
    X = _learners()
    seg = TrajectorySegmenter().fit(X, sample=400)
    again = TrajectorySegmenter.from_dict(seg.to_dict())
    assert (seg.predict(X) == again.predict(X)).all()


def test_stored_model_holds_no_learner_rows():
    seg = TrajectorySegmenter().fit(_learners(), sample=400).to_dict()
    assert set(seg) == {"weeks", "k", "seed", "reference", "centres", "names", "silhouette", "description"}
    assert len(seg["centres"]) == 4


def test_unexpected_shapes_are_refused():
    with pytest.raises(ValueError):
        TrajectorySegmenter._name(np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3.0]]))

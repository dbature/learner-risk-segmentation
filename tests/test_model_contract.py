"""The model's feature contract: no leakage, no protected attribute as input."""
from src.features.build_features import FORBIDDEN_FEATURES
from src.models.data import FEATURES, PROTECTED


def test_no_leaky_column_is_a_model_feature():
    assert not FORBIDDEN_FEATURES & set(FEATURES)


def test_protected_attributes_are_audited_not_used():
    assert not set(PROTECTED) & set(FEATURES)


def test_presentation_is_not_a_feature():
    """code_presentation is the calendar; using it would let the model learn the year."""
    assert "code_presentation" not in FEATURES

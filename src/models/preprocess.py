"""Shared preprocessing, kept free of heavy imports so the API and its tests
load it without MLflow or XGBoost installed."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.data import CATEGORICAL, NUMERIC


def preprocessor(numeric: list[str] = NUMERIC, categorical: list[str] = CATEGORICAL) -> ColumnTransformer:
    """Median imputation with explicit missing-value indicators, so 'no
    assessment due by day 30' stays visible to the model rather than being
    silently filled; scaling for the linear model; one-hot for the module."""
    num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)],
                             verbose_feature_names_out=False)

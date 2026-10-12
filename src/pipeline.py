"""Leakage-safe sklearn pipeline constructor."""
from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .data import replace_invalid_zeros
from .features import EngineeredFeaturesTransformer, FeatureSetSelector
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class ZeroToNaNTransformer(TransformerMixin, BaseEstimator):
    """Stateless conversion of selected zero sentinels to NaN."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "ZeroToNaNTransformer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ZeroToNaNTransformer requires a pandas DataFrame.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return replace_invalid_zeros(X)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        if input_features is None:
            input_features = self.feature_names_in_
        return np.asarray(input_features, dtype=object)


def build_pipeline(
    estimator: BaseEstimator,
    *,
    feature_config: str = "10",
    scale_features: bool = False,
) -> Pipeline:
    """Build a pipeline whose fitted imputer/scaler learn from training data only.

    Use ``scale_features=True`` for distance/margin/gradient-sensitive models
    (KNN, SVM, ANN) and ``False`` for the tree-based models in this project.
    """
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    # Pandas outputs let the feature engineering step work by column name.
    imputer.set_output(transform="pandas")

    scaler: BaseEstimator
    if scale_features:
        scaler = StandardScaler()
        scaler.set_output(transform="pandas")
    else:
        scaler = "passthrough"

    return Pipeline(
        steps=[
            ("zero_to_nan", ZeroToNaNTransformer()),
            ("imputer", imputer),
            ("engineer", EngineeredFeaturesTransformer()),
            ("select_features", FeatureSetSelector(feature_config=feature_config)),
            ("scaler", scaler),
            ("model", estimator),
        ]
    )

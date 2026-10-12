"""Feature engineering and selection used by all experiment protocols."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import (
    ENGINEERED_COLUMNS,
    FEATURE_CONFIGS,
    F9_COLUMN,
    F10_COLUMN,
)


class EngineeredFeaturesTransformer(TransformerMixin, BaseEstimator):
    """Add the paper's two binary features after missing-value imputation.

    The paper does not specify whether thresholds are evaluated before or after
    imputation. This implementation's explicit choice is *after* a train-fitted
    median imputer, so a missing BP/Glucose value is never silently treated as 0.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "EngineeredFeaturesTransformer":
        self._require_frame(X)
        needed = {"BloodPressure", "Glucose"}
        absent = needed.difference(X.columns)
        if absent:
            raise ValueError("Missing columns for feature engineering: " + ", ".join(sorted(absent)))
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        if any(column in X.columns for column in ENGINEERED_COLUMNS):
            raise ValueError("Input already contains engineered feature columns.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._require_frame(X)
        absent = {"BloodPressure", "Glucose"}.difference(X.columns)
        if absent:
            raise ValueError("Missing columns for feature engineering: " + ", ".join(sorted(absent)))
        if X[["BloodPressure", "Glucose"]].isna().any().any():
            raise ValueError(
                "F9/F10 require imputed BloodPressure and Glucose values. "
                "Place this transformer after the imputer in the pipeline."
            )

        result = X.copy()
        blood_pressure_high = result["BloodPressure"].gt(80)
        glucose_high = result["Glucose"].gt(105)
        result[F9_COLUMN] = (blood_pressure_high & glucose_high).astype("int64")
        result[F10_COLUMN] = blood_pressure_high.astype("int64")
        return result

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        if input_features is None:
            input_features = self.feature_names_in_
        return np.asarray([*input_features, *ENGINEERED_COLUMNS], dtype=object)

    @staticmethod
    def _require_frame(X: object) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("EngineeredFeaturesTransformer requires a pandas DataFrame.")


class FeatureSetSelector(TransformerMixin, BaseEstimator):
    """Select the paper-inspired 6-, 8-, or 10-feature configuration."""

    def __init__(self, feature_config: str = "10") -> None:
        self.feature_config = feature_config

    def fit(self, X: pd.DataFrame, y: object = None) -> "FeatureSetSelector":
        self._require_frame(X)
        if self.feature_config not in FEATURE_CONFIGS:
            raise ValueError(
                f"Unknown feature configuration {self.feature_config!r}; "
                f"choose one of {tuple(FEATURE_CONFIGS)}."
            )
        wanted = FEATURE_CONFIGS[self.feature_config]
        absent = [column for column in wanted if column not in X.columns]
        if absent:
            raise ValueError("Required features are missing: " + ", ".join(absent))
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.selected_features_ = np.asarray(wanted, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._require_frame(X)
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("FeatureSetSelector must be fitted before transform.")
        return X.loc[:, list(self.selected_features_)].copy()

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        if hasattr(self, "selected_features_"):
            return self.selected_features_.copy()
        return np.asarray(FEATURE_CONFIGS[self.feature_config], dtype=object)

    @staticmethod
    def _require_frame(X: object) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureSetSelector requires a pandas DataFrame.")

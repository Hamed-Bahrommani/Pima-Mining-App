from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.config import FEATURE_CONFIGS, FEATURE_COLUMNS, F9_COLUMN, F10_COLUMN
from src.features import EngineeredFeaturesTransformer, FeatureSetSelector
from src.pipeline import build_pipeline
from .test_data import sample_frame


def test_engineered_features_truth_table() -> None:
    frame = pd.DataFrame(
        {
            "Pregnancies": [0, 1, 2, 3, 4],
            "Glucose": [106.0, 106.0, 105.0, 200.0, 90.0],
            "BloodPressure": [81.0, 80.0, 81.0, 120.0, 60.0],
            "SkinThickness": [10.0] * 5,
            "Insulin": [50.0] * 5,
            "BMI": [24.0] * 5,
            "DiabetesPedigreeFunction": [0.3] * 5,
            "Age": [30] * 5,
        }
    )
    result = EngineeredFeaturesTransformer().fit_transform(frame)
    assert result[F9_COLUMN].tolist() == [1, 0, 0, 1, 0]
    assert result[F10_COLUMN].tolist() == [1, 0, 1, 1, 0]



def test_engineering_rejects_unimputed_threshold_inputs() -> None:
    frame = pd.DataFrame({"Glucose": [np.nan], "BloodPressure": [80.0]})
    transformer = EngineeredFeaturesTransformer().fit(frame)
    with pytest.raises(ValueError, match="after the imputer"):
        transformer.transform(frame)


@pytest.mark.parametrize("config", ["6", "8", "10"])
def test_feature_selector_returns_expected_width(config: str) -> None:
    data = sample_frame().loc[:, FEATURE_COLUMNS].copy()
    # Impute values for the selector's standalone test.
    data = data.fillna(data.median())
    engineered = EngineeredFeaturesTransformer().fit_transform(data)
    selected = FeatureSetSelector(feature_config=config).fit_transform(engineered)
    assert selected.shape[1] == int(config)
    assert tuple(selected.columns) == FEATURE_CONFIGS[config]


def test_pipeline_fits_medians_on_training_rows_only() -> None:
    frame = sample_frame(24)
    X = frame.loc[:, FEATURE_COLUMNS].copy()
    y = frame["Outcome"].copy()
    X_train, X_test = X.iloc[:18].copy(), X.iloc[18:].copy()
    # An extreme held-out value must not influence the fitted scaler/imputer.
    X_test.loc[X_test.index[0], "Glucose"] = 9999

    pipeline = build_pipeline(DummyClassifier(strategy="most_frequent"), feature_config="10")
    pipeline.fit(X_train, y.iloc[:18])

    from src.data import replace_invalid_zeros
    expected = replace_invalid_zeros(X_train).median(skipna=True).to_numpy(dtype=float)
    observed = pipeline.named_steps["imputer"].statistics_
    np.testing.assert_allclose(observed, expected)

    transformed = pipeline[:-1].transform(X_test)
    assert transformed.shape[1] == 10
    assert not transformed.isna().any().any()

@pytest.mark.parametrize("config", ["6", "8", "10"])
@pytest.mark.parametrize("scale", [False, True])
def test_pipeline_runs_for_each_feature_configuration(config: str, scale: bool) -> None:
    frame = sample_frame(24)
    X = frame.loc[:, FEATURE_COLUMNS]
    y = frame["Outcome"]
    pipeline = build_pipeline(
        DummyClassifier(strategy="most_frequent"),
        feature_config=config,
        scale_features=scale,
    )
    pipeline.fit(X.iloc[:18], y.iloc[:18])
    predictions = pipeline.predict(X.iloc[18:])
    assert len(predictions) == 6
    assert set(predictions).issubset({0, 1})


def test_impossible_zero_values_are_imputed_not_left_as_zero() -> None:
    frame = sample_frame(24)
    X = frame.loc[:, FEATURE_COLUMNS]
    y = frame["Outcome"]
    pipeline = build_pipeline(DummyClassifier(strategy="most_frequent"), feature_config="8")
    pipeline.fit(X.iloc[:18], y.iloc[:18])
    transformed = pipeline[:-1].transform(X.iloc[18:])
    for column in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        assert not transformed[column].eq(0).any()


def test_missing_test_value_is_filled_with_training_median() -> None:
    frame = sample_frame(24)
    X = frame.loc[:, FEATURE_COLUMNS].copy()
    y = frame["Outcome"]
    X_train, X_test = X.iloc[:18].copy(), X.iloc[18:].copy()

    # Test rows look nothing like training rows, except one invalid zero.
    X_test["Glucose"] = 250
    X_test.iloc[0, X_test.columns.get_loc("Glucose")] = 0

    pipeline = build_pipeline(DummyClassifier(strategy="most_frequent"), feature_config="8")
    pipeline.fit(X_train, y.iloc[:18])
    transformed = pipeline[:-1].transform(X_test)

    train_median = X_train["Glucose"].replace(0, np.nan).median()
    assert transformed["Glucose"].iloc[0] == train_median
    assert transformed["Glucose"].iloc[1] == 250  # valid test values untouched


def test_imputer_statistics_do_not_depend_on_outcome_labels() -> None:
    frame = sample_frame(24)
    X = frame.loc[:, FEATURE_COLUMNS]
    y = frame["Outcome"]

    original = build_pipeline(DummyClassifier(strategy="most_frequent"), feature_config="10")
    flipped = build_pipeline(DummyClassifier(strategy="most_frequent"), feature_config="10")
    original.fit(X, y)
    flipped.fit(X, 1 - y)

    np.testing.assert_array_equal(
        original.named_steps["imputer"].statistics_,
        flipped.named_steps["imputer"].statistics_,
    )


def test_feature_configurations_match_the_paper_definition() -> None:
    # Hard-coded on purpose: this must not be derived from FEATURE_CONFIGS itself.
    original_eight = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
    ]
    six = ["Pregnancies", "Glucose", "BloodPressure", "Insulin", "BMI", "Age"]

    assert set(FEATURE_CONFIGS) == {"6", "8", "10"}
    assert list(FEATURE_CONFIGS["6"]) == six
    assert list(FEATURE_CONFIGS["8"]) == original_eight
    assert list(FEATURE_CONFIGS["10"]) == original_eight + [F9_COLUMN, F10_COLUMN]
    for columns in FEATURE_CONFIGS.values():
        assert "Outcome" not in columns
        assert len(columns) == len(set(columns))


def test_unknown_feature_configuration_is_rejected() -> None:
    frame = sample_frame().loc[:, FEATURE_COLUMNS]
    with pytest.raises(ValueError, match="Unknown feature configuration"):
        FeatureSetSelector(feature_config="7").fit(frame)


def test_engineered_features_use_imputed_values_not_raw_zeros() -> None:
    frame = sample_frame(24)
    X = frame.loc[:, FEATURE_COLUMNS].copy()
    y = frame["Outcome"]
    X_train, X_test = X.iloc[:18].copy(), X.iloc[18:].copy()

    X_train["BloodPressure"] = 90  # training median becomes 90
    X_train.iloc[0, X_train.columns.get_loc("BloodPressure")] = 0  # one invalid zero
    X_test["Glucose"] = 200
    X_test.iloc[0, X_test.columns.get_loc("BloodPressure")] = 0  # missing BP in test row

    pipeline = build_pipeline(DummyClassifier(strategy="most_frequent"), feature_config="10")
    pipeline.fit(X_train, y.iloc[:18])
    transformed = pipeline[:-1].transform(X_test)

    assert transformed["BloodPressure"].iloc[0] == 90
    assert transformed[F9_COLUMN].iloc[0] == 1
    assert transformed[F10_COLUMN].iloc[0] == 1


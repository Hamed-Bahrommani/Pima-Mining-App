"""Integration tests that run against the real Pima CSV.

They are skipped automatically when the CSV is not present, so the offline
unit tests never depend on a download.
"""
from __future__ import annotations

import pytest
from sklearn.dummy import DummyClassifier

from src.config import DATA_PATH, F9_COLUMN, F10_COLUMN, ZERO_AS_MISSING_COLUMNS
from src.data import load_dataset, sha256_file, split_features_target
from src.pipeline import build_pipeline

pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason="Real Pima CSV not downloaded (run python scripts/fetch_data.py)",
)

# Calculated from the real file during Day 2 validation (23,278 bytes, LF endings).
EXPECTED_SHA256 = "6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af"

# Observed in our file; these match the commonly reported figures for this dataset.
OBSERVED_ZERO_COUNTS = {
    "Pregnancies": 111,
    "Glucose": 5,
    "BloodPressure": 35,
    "SkinThickness": 227,
    "Insulin": 374,
    "BMI": 11,
    "DiabetesPedigreeFunction": 0,
    "Age": 0,
}


def test_real_csv_hash_matches_recorded_values() -> None:
    assert sha256_file(DATA_PATH) == EXPECTED_SHA256
    checksum_file = DATA_PATH.with_suffix(DATA_PATH.suffix + ".sha256")
    if checksum_file.exists():
        assert checksum_file.read_text().split()[0] == EXPECTED_SHA256


def test_real_csv_shape_and_class_counts() -> None:
    frame = load_dataset()
    assert frame.shape == (768, 9)
    assert frame["Outcome"].value_counts().to_dict() == {0: 500, 1: 268}


def test_loader_leaves_raw_zeros_untouched() -> None:
    X, _ = split_features_target(load_dataset())
    assert (X == 0).sum().to_dict() == OBSERVED_ZERO_COUNTS


def test_pipeline_removes_invalid_zeros_but_keeps_zero_pregnancies() -> None:
    X, y = split_features_target(load_dataset())
    # Fitting on all rows is fine here: this checks a data property, not performance.
    pipeline = build_pipeline(DummyClassifier(strategy="most_frequent"), feature_config="8")
    pipeline.fit(X, y)
    transformed = pipeline[:-1].transform(X)

    assert transformed.shape == (768, 8)
    assert not transformed.isna().any().any()
    for column in ZERO_AS_MISSING_COLUMNS:
        assert not transformed[column].eq(0).any()
    assert transformed["Pregnancies"].eq(0).sum() == OBSERVED_ZERO_COUNTS["Pregnancies"]


def test_real_data_engineered_features_are_consistent() -> None:
    X, y = split_features_target(load_dataset())
    pipeline = build_pipeline(DummyClassifier(strategy="most_frequent"), feature_config="10")
    pipeline.fit(X, y)
    transformed = pipeline[:-1].transform(X)

    assert transformed.shape == (768, 10)
    assert set(transformed[F9_COLUMN].unique()) <= {0, 1}
    assert (transformed[F9_COLUMN] <= transformed[F10_COLUMN]).all()  # F9 implies F10
    assert (transformed[F10_COLUMN] == (transformed["BloodPressure"] > 80)).all()


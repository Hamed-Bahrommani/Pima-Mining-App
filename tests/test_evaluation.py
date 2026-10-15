"""Tests for evaluation and cross-validation helpers."""

import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.evaluation import cross_validate_experiment, evaluate_predictions


@pytest.fixture
def small_dataset():
    rows = 100
    X = pd.DataFrame({
        column: [
            float((row * (i + 3)) % 120 + 1)
            for row in range(rows)
        ]
        for i, column in enumerate(FEATURE_COLUMNS)
    })
    y = pd.Series([0, 1] * (rows // 2), name=TARGET_COLUMN)
    return X, y


def test_evaluate_predictions_returns_expected_counts():
    result = evaluate_predictions([0, 0, 1, 1], [0, 1, 1, 0])

    assert result["true_negatives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["true_positives"] == 1
    assert result["accuracy"] == 0.5
    assert result["specificity"] == 0.5


def test_cross_validation_returns_mean_and_std(small_dataset):
    X, y = small_dataset
    result = cross_validate_experiment(
        X, y,
        model_name="random_forest",
        feature_config="10",
        folds=3,
    )

    assert result["model"] == "random_forest"
    assert result["feature_config"] == "10"

    for metric in ("accuracy", "precision_positive", "recall_positive", "f1_positive"):
        assert 0 <= result[f"{metric}_mean"] <= 1
        assert result[f"{metric}_std"] >= 0


def test_cross_validation_rejects_unknown_model(small_dataset):
    X, y = small_dataset

    with pytest.raises(ValueError, match="Unknown model"):
        cross_validate_experiment(X, y, model_name="not_a_model")


def test_cross_validation_rejects_invalid_fold_count(small_dataset):
    X, y = small_dataset

    with pytest.raises(ValueError, match="folds"):
        cross_validate_experiment(X, y, folds=1)

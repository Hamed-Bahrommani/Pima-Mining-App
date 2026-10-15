"""Tests for reproducible model experiments."""

import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.experiments import run_experiment


@pytest.fixture
def small_dataset():
    """Create a deterministic dataset with all required predictor columns."""
    rows = 100
    data = {}

    for i, column in enumerate(FEATURE_COLUMNS):
        data[column] = [
            float((row * (i + 3)) % 120 + 1)
            for row in range(rows)
        ]

    # Avoid sentinel zeros in the synthetic data.
    X = pd.DataFrame(data)
    y = pd.Series([0, 1] * (rows // 2), name=TARGET_COLUMN)

    return X, y


def test_experiment_returns_metrics_and_expected_split(small_dataset):
    X, y = small_dataset

    result = run_experiment(
        X,
        y,
        model_name="random_forest",
        feature_config="10",
    )

    assert result["train_rows"] == 70
    assert result["test_rows"] == 30

    for metric in (
        "accuracy",
        "precision_positive",
        "recall_positive",
        "f1_positive",
    ):
        assert 0 <= result[metric] <= 1

    assert result["pipeline"] is not None


def test_experiment_rejects_unknown_model(small_dataset):
    X, y = small_dataset

    with pytest.raises(ValueError, match="Unknown model"):
        run_experiment(X, y, model_name="not_a_model")


def test_experiment_rejects_invalid_test_size(small_dataset):
    X, y = small_dataset

    with pytest.raises(ValueError, match="test_size"):
        run_experiment(X, y, test_size=1.0)


def test_experiment_rejects_mismatched_row_counts(small_dataset):
    X, y = small_dataset

    with pytest.raises(ValueError, match="same number of rows"):
        run_experiment(X, y.iloc[:-1])



def test_compare_experiments_returns_requested_combinations(small_dataset):
    from src.experiments import compare_experiments

    X, y = small_dataset
    result = compare_experiments(
        X,
        y,
        feature_configs=("6", "10"),
        model_names=("knn", "random_forest"),
        folds=3,
    )

    assert len(result) == 4
    assert set(result["feature_config"]) == {"6", "10"}
    assert set(result["model"]) == {"knn", "random_forest"}

    assert "accuracy_mean" in result.columns
    assert "accuracy_std" in result.columns
    assert "test_accuracy" in result.columns


def test_compare_experiments_rejects_unknown_model(small_dataset):
    from src.experiments import compare_experiments

    X, y = small_dataset

    with pytest.raises(ValueError, match="Unknown models"):
        compare_experiments(X, y, model_names=("not_a_model",))

        

"""Tests for reproducible model experiments."""

import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.experiments import run_experiment
import src.experiments as experiments
from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE


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

        

def test_compare_experiments_keeps_test_partition_out_of_cv(
    small_dataset, monkeypatch
):
    X, y = small_dataset

    expected_X_train, expected_X_test, expected_y_train, expected_y_test = (
        train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    cv_calls = []
    test_partitions = []

    def fake_cross_validate(
        X_cv,
        y_cv,
        *,
        model_name,
        feature_config,
        folds,
        tune_hyperparameters=False,
        estimator_params=None,
    ):
        cv_calls.append((set(X_cv.index), set(y_cv.index)))
        return {
            "model": model_name,
            "feature_config": str(feature_config),
            "accuracy_mean": 0.75,
            "accuracy_std": 0.01,
        }

    original_evaluate_split = experiments._evaluate_split

    def recording_evaluate_split(
        X_train,
        X_test,
        y_train,
        y_test,
        **kwargs,
    ):
        test_partitions.append(
            (set(X_train.index), set(X_test.index))
        )
        return original_evaluate_split(
            X_train,
            X_test,
            y_train,
            y_test,
            **kwargs,
        )

    monkeypatch.setattr(
        experiments, "cross_validate_experiment", fake_cross_validate
    )
    monkeypatch.setattr(
        experiments, "_evaluate_split", recording_evaluate_split
    )

    result = experiments.compare_experiments(
        X,
        y,
        feature_configs=("6", "10"),
        model_names=("knn", "random_forest"),
        folds=3,
    )

    expected_train_indices = set(expected_X_train.index)
    expected_test_indices = set(expected_X_test.index)

    assert len(result) == 4
    assert len(cv_calls) == 4
    assert len(test_partitions) == 4

    # Every CV call receives training rows only.
    for X_indices, y_indices in cv_calls:
        assert X_indices == expected_train_indices
        assert y_indices == set(expected_y_train.index)
        assert X_indices.isdisjoint(expected_test_indices)

    # Every model/configuration uses the same held-out test rows.
    for train_indices, test_indices in test_partitions:
        assert train_indices == expected_train_indices
        assert test_indices == expected_test_indices
        assert train_indices.isdisjoint(test_indices)

    assert set(expected_y_test.index) == expected_test_indices


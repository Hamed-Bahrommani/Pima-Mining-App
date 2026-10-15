"""Reproducible train/test experiments for the diabetes classifiers."""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE
from src.models import get_models, model_requires_scaling
from src.pipeline import build_pipeline
from src.data import load_dataset
from src.config import TARGET_COLUMN

from src.evaluation import cross_validate_experiment


def run_experiment(
    X,
    y,
    *,
    model_name="random_forest",
    feature_config="10",
    test_size=0.30,
):
    """Fit one classifier and evaluate it on a stratified held-out test set.

    X must contain original predictor columns only, and y the Outcome labels.
    Preprocessing is fitted on training rows only.
    """
    models = get_models()

    if model_name not in models:
        raise ValueError(
            f"Unknown model {model_name!r}; choose one of {tuple(models)}."
        )
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline(
        estimator=models[model_name],
        feature_config=feature_config,
        scale_features=model_requires_scaling(model_name),
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    return {
        "model": model_name,
        "feature_config": str(feature_config),
        "test_size": test_size,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": accuracy_score(y_test, predictions),
        "precision_positive": precision_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "recall_positive": recall_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "f1_positive": f1_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "pipeline": pipeline,
    }


def run_default_experiment(
    *,
    model_name="random_forest",
    feature_config="10",
    test_size=0.30,
):
    """Load the project dataset and run one experiment with standard settings."""
    dataset = load_dataset()
    X = dataset.drop(columns=[TARGET_COLUMN])
    y = dataset[TARGET_COLUMN]

    return run_experiment(
        X,
        y,
        model_name=model_name,
        feature_config=feature_config,
        test_size=test_size,
    )


def compare_experiments(
    X,
    y,
    *,
    feature_configs=("6", "8", "10"),
    model_names=None,
    test_size=0.30,
    folds=5,
):
    """Compare models and feature sets using CV and held-out test metrics."""
    import pandas as pd

    if model_names is None:
        model_names = tuple(get_models())

    available_models = get_models()
    unknown_models = set(model_names) - set(available_models)
    if unknown_models:
        raise ValueError(f"Unknown models: {sorted(unknown_models)}")

    results = []

    for feature_config in feature_configs:
        for model_name in model_names:
            cv_result = cross_validate_experiment(
                X,
                y,
                model_name=model_name,
                feature_config=feature_config,
                folds=folds,
            )

            test_result = run_experiment(
                X,
                y,
                model_name=model_name,
                feature_config=feature_config,
                test_size=test_size,
            )

            row = {
                **cv_result,
                **{
                    f"test_{key}": value
                    for key, value in test_result.items()
                    if key not in {"pipeline", "model", "feature_config", "test_size"}
                },
            }
            results.append(row)

    return pd.DataFrame(results)



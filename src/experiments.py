"""Reproducible train/test experiments for the diabetes classifiers."""


"""Reproducible train/test experiments for the diabetes classifiers."""

from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TARGET_COLUMN
from src.models import get_models, model_requires_scaling
from src.pipeline import build_pipeline
from src.data import load_dataset
from src.evaluation import (
    cross_validate_experiment,
    evaluate_predictions,
    get_positive_scores,
    select_f1_threshold,
    tune_experiment,
)





def _evaluate_split(
    X_train,
    X_test,
    y_train,
    y_test,
    *,
    model_name,
    feature_config,
    test_size,
    estimator_params=None,
):
    """Fit on training rows, select a threshold there, and evaluate held-out rows."""
    models = get_models()
    pipeline = build_pipeline(
        estimator=models[model_name],
        feature_config=feature_config,
        scale_features=model_requires_scaling(model_name),
    )

    if estimator_params:
        pipeline.set_params(**estimator_params)

    threshold_result = select_f1_threshold(
        X_train,
        y_train,
        estimator=pipeline,
        folds=5,
    )

    pipeline.fit(X_train, y_train)
    test_scores = get_positive_scores(pipeline, X_test)
    predictions = (
        test_scores >= threshold_result["threshold"]
    ).astype(int)

    return {
        "model": model_name,
        "feature_config": str(feature_config),
        "test_size": test_size,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        **evaluate_predictions(y_test, predictions),
        "threshold": threshold_result["threshold"],
        "threshold_training_oof_f1": threshold_result["training_oof_f1"],
        "score_method": threshold_result["score_method"],
        "pipeline": pipeline,
    }


def run_experiment(
    X,
    y,
    *,
    model_name="random_forest",
    feature_config="10",
    test_size=0.30,
):
    """Fit one classifier and evaluate it on a stratified held-out test set."""
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

    return _evaluate_split(
        X_train,
        X_test,
        y_train,
        y_test,
        model_name=model_name,
        feature_config=feature_config,
        test_size=test_size,
    )



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
    """Compare classifiers using training-only tuning and threshold selection."""
    import pandas as pd

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")
    if folds < 2:
        raise ValueError("folds must be at least 2.")

    available_models = get_models()
    if model_names is None:
        model_names = tuple(available_models)

    unknown_models = set(model_names) - set(available_models)
    if unknown_models:
        raise ValueError(f"Unknown models: {sorted(unknown_models)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results = []

    for feature_config in feature_configs:
        for model_name in model_names:
            # Each outer CV fold performs its own training-only threshold
            # selection; RF and ANN also tune within each outer training fold.
            cv_result = cross_validate_experiment(
                X_train,
                y_train,
                model_name=model_name,
                feature_config=feature_config,
                folds=folds,
                tune_hyperparameters=model_name in {"random_forest", "ann"},
            )

            estimator_params = None

            if model_name in {"random_forest", "ann"}:
                tuned_model = tune_experiment(
                    X_train,
                    y_train,
                    model_name=model_name,
                    feature_config=feature_config,
                    folds=folds,
                )
                estimator_params = tuned_model["best_params"]

            test_result = _evaluate_split(
                X_train,
                X_test,
                y_train,
                y_test,
                model_name=model_name,
                feature_config=feature_config,
                test_size=test_size,
                estimator_params=estimator_params,
            )

            if model_name in {"random_forest", "ann"}:
                test_result["best_params"] = estimator_params

            row = {
                **cv_result,
                **{
                    f"test_{key}": value
                    for key, value in test_result.items()
                    if key not in {"pipeline", "model", "feature_config"}
                },
            }
            results.append(row)

    return pd.DataFrame(results)


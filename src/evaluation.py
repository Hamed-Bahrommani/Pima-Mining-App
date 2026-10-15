"""Evaluation helpers for diabetes classification experiments."""

import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import RANDOM_STATE
from src.models import get_models, model_requires_scaling
from src.pipeline import build_pipeline


def cross_validate_experiment(
    X,
    y,
    *,
    model_name="random_forest",
    feature_config="10",
    folds=5,
):
    """Evaluate one model with stratified cross-validation."""
    models = get_models()

    if model_name not in models:
        raise ValueError(
            f"Unknown model {model_name!r}; choose one of {tuple(models)}."
        )
    if folds < 2:
        raise ValueError("folds must be at least 2.")

    pipeline = build_pipeline(
        estimator=models[model_name],
        feature_config=feature_config,
        scale_features=model_requires_scaling(model_name),
    )

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision_positive": "precision",
            "recall_positive": "recall",
            "f1_positive": "f1",
        },
        return_train_score=False,
        error_score="raise",
    )

    results = {"model": model_name, "feature_config": str(feature_config)}
    for metric in (
        "accuracy",
        "precision_positive",
        "recall_positive",
        "f1_positive",
    ):
        values = scores[f"test_{metric}"]
        results[f"{metric}_mean"] = values.mean()
        results[f"{metric}_std"] = values.std()

    return results


def evaluate_predictions(y_true, y_pred):
    """Calculate test metrics and a confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_positive": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall_positive": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1_positive": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


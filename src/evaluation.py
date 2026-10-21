
"""Evaluation helpers for diabetes classification experiments."""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
)

from src.config import RANDOM_STATE
from src.models import get_models, model_requires_scaling
from src.pipeline import build_pipeline


def _take_rows(data, indices):
    """Select rows from pandas or array-like data."""
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return np.asarray(data)[indices]


def _score_method(estimator):
    """Return the preferred method for obtaining positive-class scores."""
    if hasattr(estimator, "predict_proba"):
        return "predict_proba"
    if hasattr(estimator, "decision_function"):
        return "decision_function"
    raise ValueError(
        "Threshold optimization requires predict_proba or decision_function."
    )


def _extract_positive_scores(raw_scores, classes, method):
    """Convert estimator scores to scores where larger means more likely class 1."""
    classes = np.asarray(classes)

    positive_indices = np.flatnonzero(classes == 1)
    if len(positive_indices) != 1:
        raise ValueError("The target must contain the positive class label 1.")

    if method == "predict_proba":
        return np.asarray(raw_scores)[:, positive_indices[0]]

    scores = np.asarray(raw_scores).reshape(-1)

    # Binary decision_function scores are positive toward classes_[1].
    if len(classes) != 2:
        raise ValueError("Decision-function thresholding requires binary classes.")
    if classes[1] != 1:
        scores = -scores

    return scores


def get_positive_scores(estimator, X):
    """Get fitted-estimator scores where higher values favor positive class 1."""
    method = _score_method(estimator)
    classes = getattr(estimator, "classes_", None)

    if classes is None:
        raise ValueError("The estimator must be fitted before scoring.")

    raw_scores = getattr(estimator, method)(X)
    return _extract_positive_scores(raw_scores, classes, method)


def select_f1_threshold(
    X,
    y,
    *,
    estimator,
    folds=5,
):
    """Select a positive-class F1 threshold from training-only OOF scores.

    For predict_proba estimators, the threshold is a probability threshold.
    For decision_function estimators, it is a decision-score threshold.

    Ties are resolved deterministically: probability thresholds closest to 0.5
    are preferred; remaining ties prefer the higher threshold. For decision
    scores, the highest threshold among tied candidates is preferred.
    """
    y_array = np.asarray(y)

    if folds < 2:
        raise ValueError("folds must be at least 2.")
    if len(X) != len(y_array):
        raise ValueError("X and y must contain the same number of rows.")
    if set(np.unique(y_array)) != {0, 1}:
        raise ValueError("Threshold selection requires binary labels 0 and 1.")

    class_counts = np.unique(y_array, return_counts=True)[1]
    if class_counts.min() < folds:
        raise ValueError(
            "The number of folds cannot exceed the smallest class count."
        )

    method = _score_method(estimator)
    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    raw_scores = cross_val_predict(
        estimator,
        X,
        y_array,
        cv=cv,
        method=method,
    )
    scores = _extract_positive_scores(
        raw_scores,
        np.unique(y_array),
        method,
    )

    # Include a threshold above the maximum score, allowing an all-negative
    # prediction if that is optimal.
    thresholds = np.unique(
        np.append(scores, np.nextafter(scores.max(), np.inf))
    )

    scored_thresholds = []
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        f1 = f1_score(y_array, predictions, zero_division=0)
        scored_thresholds.append((float(threshold), float(f1)))

    best_f1 = max(score for _, score in scored_thresholds)
    best_thresholds = [
        threshold
        for threshold, score in scored_thresholds
        if np.isclose(score, best_f1)
    ]

    if method == "predict_proba":
        threshold = min(
            best_thresholds,
            key=lambda value: (abs(value - 0.5), -value),
        )
    else:
        threshold = max(best_thresholds)

    return {
        "threshold": float(threshold),
        "training_oof_f1": float(best_f1),
        "score_method": method,
    }


def evaluate_predictions(y_true, y_pred):
    """Calculate classification metrics and a confusion matrix."""
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


def cross_validate_experiment(
    X,
    y,
    *,
    model_name="random_forest",
    feature_config="10",
    folds=5,
    estimator_params=None,
    tune_hyperparameters=False,
):
    """Estimate performance using outer CV and training-only threshold selection."""
    y_array = np.asarray(y)
    models = get_models()

    if model_name not in models:
        raise ValueError(
            f"Unknown model {model_name!r}; choose one of {tuple(models)}."
        )
    if folds < 2:
        raise ValueError("folds must be at least 2.")
    if len(X) != len(y_array):
        raise ValueError("X and y must contain the same number of rows.")
    if set(np.unique(y_array)) != {0, 1}:
        raise ValueError("Cross-validation requires binary labels 0 and 1.")

    class_counts = np.unique(y_array, return_counts=True)[1]
    if class_counts.min() < folds:
        raise ValueError(
            "The number of folds cannot exceed the smallest class count."
        )

    outer_cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_metrics = []

    for train_indices, validation_indices in outer_cv.split(X, y_array):
        X_train = _take_rows(X, train_indices)
        y_train = y_array[train_indices]
        X_validation = _take_rows(X, validation_indices)
        y_validation = y_array[validation_indices]

        pipeline = build_pipeline(
            estimator=models[model_name],
            feature_config=feature_config,
            scale_features=model_requires_scaling(model_name),
        )

        if estimator_params:
            pipeline.set_params(**estimator_params)

        # Hyperparameter selection, when applicable, is confined to the
        # current outer-training fold.
        if (
                tune_hyperparameters
                and model_name in {"random_forest", "ann"}
                and not estimator_params
            ):
            tuned = tune_experiment(
                X_train,
                y_train,
                model_name=model_name,
                feature_config=feature_config,
                folds=min(folds, 3),
            )
            pipeline = tuned["best_estimator"]

        threshold_result = select_f1_threshold(
            X_train,
            y_train,
            estimator=pipeline,
            folds=min(folds, 5),
        )

        pipeline.fit(X_train, y_train)

        validation_scores = get_positive_scores(pipeline, X_validation)
        validation_predictions = (
            validation_scores >= threshold_result["threshold"]
        ).astype(int)

        fold_metrics.append(
            evaluate_predictions(y_validation, validation_predictions)
        )

    results = {
        "model": model_name,
        "feature_config": str(feature_config),
    }

    for metric in (
        "accuracy",
        "precision_positive",
        "recall_positive",
        "f1_positive",
    ):
        values = np.array([fold[metric] for fold in fold_metrics])
        results[f"{metric}_mean"] = float(values.mean())
        results[f"{metric}_std"] = float(values.std())

    return results


def tune_experiment(
    X,
    y,
    *,
    model_name="random_forest",
    feature_config="10",
    folds=5,
):
    """Select hyperparameters using stratified cross-validation."""
    if model_name not in {"random_forest", "ann"}:
        raise ValueError(
            "Tuning is supported only for 'random_forest' and 'ann'."
        )
    if folds < 2:
        raise ValueError("folds must be at least 2.")

    models = get_models()
    pipeline = build_pipeline(
        estimator=models[model_name],
        feature_config=feature_config,
        scale_features=model_requires_scaling(model_name),
    )

    if model_name == "random_forest":
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10],
            "model__min_samples_leaf": [1, 2],
        }
    else:
        param_grid = {
            "model__hidden_layer_sizes": [(50,), (50, 25)],
            "model__alpha": [0.001, 0.01],
            "model__learning_rate_init": [0.001],
        }

    cv = StratifiedKFold(
    n_splits=min(folds, 3),
    shuffle=True,
    random_state=RANDOM_STATE,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring={"accuracy": "accuracy", "f1": "f1"},
        refit="f1",
        cv=cv,
        n_jobs=-1,
        error_score="raise",
    )
    search.fit(X, y)

    best_index = search.best_index_

    return {
        "model": model_name,
        "feature_config": str(feature_config),
        "best_params": search.best_params_,
        "best_cv_accuracy": search.cv_results_["mean_test_accuracy"][best_index],
        "best_cv_f1": search.best_score_,
        "best_estimator": search.best_estimator_,
    }

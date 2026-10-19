"""Compare MLP settings by convergence and CV accuracy, using training rows only."""
import warnings

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.config import RANDOM_STATE
from src.data import load_dataset, split_features_target
from src.models import get_models
from src.pipeline import build_pipeline

CANDIDATES = {
    "current": {},
    "lbfgs, 2000 iter": {"solver": "lbfgs", "max_iter": 2000},
    "adam + early stopping": {"early_stopping": True, "n_iter_no_change": 20},
    "adam, one small layer (16)": {"hidden_layer_sizes": (16,)},
    "adam, alpha=0.01, tol=1e-3": {"alpha": 0.01, "tol": 1e-3},
}


def main() -> None:
    X, y = split_features_target(load_dataset())
    X_train, _X_test, y_train, _y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )  # the test rows are never touched here
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    base = get_models()["ann"]
    print("current ANN params:", base.get_params())

    for label, params in CANDIDATES.items():
        for config in ("6", "8", "10"):
            estimator = clone(base).set_params(**params)
            pipeline = build_pipeline(estimator, feature_config=config, scale_features=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
                pipeline.fit(X_train, y_train)
            unconverged = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
            print(
                f"{label:30s} features={config:>2s} "
                f"unconverged_fits={unconverged}/6 "
                f"iterations={pipeline[-1].n_iter_:5d} "
                f"cv_accuracy={scores.mean():.3f}"
            )


main()
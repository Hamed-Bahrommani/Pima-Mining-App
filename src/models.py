
"""Central registry of classifiers used in the diabetes experiments."""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE


def get_models():
    """Return fresh, reproducibly configured classifier instances."""
    return {
        "knn": KNeighborsClassifier(),
        "random_forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
        ),
        "svm": SVC(),
        "ann": MLPClassifier(
            random_state=RANDOM_STATE,
            alpha=0.01,
            tol=1e-3,
            max_iter=1000,
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
    }


def model_requires_scaling(model_name):
    """Whether a model should receive standardized input features."""
    if model_name not in get_models():
        raise ValueError(f"Unknown model: {model_name}")

    return model_name in {"knn", "svm", "ann"}



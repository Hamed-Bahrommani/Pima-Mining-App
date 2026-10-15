
import pytest
from sklearn.base import ClassifierMixin

from src.models import get_models, model_requires_scaling


def test_registry_contains_expected_models():
    assert set(get_models()) == {
        "knn",
        "random_forest",
        "svm",
        "ann",
        "decision_tree",
    }


def test_registry_returns_fresh_instances():
    first = get_models()
    second = get_models()

    for name in first:
        assert first[name] is not second[name]


def test_registry_models_are_classifiers():
    for model in get_models().values():
        assert isinstance(model, ClassifierMixin)


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("knn", True),
        ("random_forest", False),
        ("svm", True),
        ("ann", True),
        ("decision_tree", False),
    ],
)
def test_model_scaling_requirements(model_name, expected):
    assert model_requires_scaling(model_name) is expected


def test_unknown_model_scaling_requirement_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        model_requires_scaling("not_a_model")


"""Leakage canaries for the clean preprocessing pipeline (offline, synthetic data)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.config import FEATURE_COLUMNS, ZERO_AS_MISSING_COLUMNS
from src.pipeline import build_pipeline

ZERO_RATES = {
    "Glucose": 0.01, "BloodPressure": 0.05, "SkinThickness": 0.30,
    "Insulin": 0.45, "BMI": 0.02,
}


def pima_like_features(n: int = 768, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "Pregnancies": rng.integers(0, 12, n),
        "Glucose": rng.normal(120, 30, n).round().clip(40, 199),
        "BloodPressure": rng.normal(70, 12, n).round().clip(24, 122),
        "SkinThickness": rng.normal(25, 9, n).round().clip(7, 99),
        "Insulin": rng.normal(120, 80, n).round().clip(15, 600),
        "BMI": rng.normal(32, 6, n).round(1).clip(18, 60),
        "DiabetesPedigreeFunction": rng.gamma(2, 0.25, n).round(3),
        "Age": rng.integers(21, 70, n),
    })
    for column, rate in ZERO_RATES.items():
        X.loc[rng.random(n) < rate, column] = 0
    return X[list(FEATURE_COLUMNS)]


def random_labels(n: int, seed: int) -> pd.Series:
    """Labels independent of the features: there is nothing real to learn."""
    return pd.Series((np.random.default_rng(seed).random(n) < 0.35).astype(int))


def leaky_class_median_impute(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Deliberately leaky (class medians from ALL rows). Only proves the canary can fail."""
    imputed = X.astype(float).copy()
    labels = y.to_numpy()
    for column in ZERO_AS_MISSING_COLUMNS:
        for label in (0, 1):
            in_class = labels == label
            values = imputed[column].to_numpy(copy=True)
            values[in_class & (values == 0)] = np.median(values[in_class & (values != 0)])
            imputed[column] = values
    return imputed


def test_preprocessing_output_does_not_depend_on_training_labels() -> None:
    X = pima_like_features()
    y = random_labels(len(X), seed=1)
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=0
    )
    shuffled = pd.Series(
        np.random.default_rng(5).permutation(y_train.to_numpy()), index=y_train.index
    )
    a = build_pipeline(RandomForestClassifier(random_state=0), feature_config="10")
    b = build_pipeline(RandomForestClassifier(random_state=0), feature_config="10")
    a.fit(X_train, y_train)
    b.fit(X_train, shuffled)
    pd.testing.assert_frame_equal(a[:-1].transform(X_test), b[:-1].transform(X_test))


def test_transform_of_a_row_does_not_depend_on_other_test_rows() -> None:
    X = pima_like_features()
    y = random_labels(len(X), seed=1)
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=0
    )
    pipeline = build_pipeline(RandomForestClassifier(random_state=0), feature_config="10")
    pipeline.fit(X_train, y_train)
    batch = pipeline[:-1].transform(X_test)
    single = pipeline[:-1].transform(X_test.iloc[[0]])
    pd.testing.assert_frame_equal(single, batch.iloc[[0]])


def test_random_labels_expose_leakage_in_leaky_imputation_but_not_in_clean_pipeline() -> None:
    X = pima_like_features()
    y = random_labels(len(X), seed=2)
    train_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=0.30, stratify=y, random_state=0
    )

    clean = build_pipeline(RandomForestClassifier(random_state=0), feature_config="8")
    clean.fit(X.iloc[train_idx], y.iloc[train_idx])
    clean_accuracy = (clean.predict(X.iloc[test_idx]) == y.iloc[test_idx]).mean()

    X_leaky = leaky_class_median_impute(X, y)
    leaky = RandomForestClassifier(random_state=0)
    leaky.fit(X_leaky.iloc[train_idx], y.iloc[train_idx])
    leaky_accuracy = (leaky.predict(X_leaky.iloc[test_idx]) == y.iloc[test_idx]).mean()

    majority_rate = max(y.mean(), 1 - y.mean())
    # Nothing real to learn, so a clean pipeline stays near the majority rate ...
    assert clean_accuracy <= majority_rate + 0.05
    # ... and the canary is only meaningful if it can detect actual leakage.
    assert leaky_accuracy >= clean_accuracy + 0.07

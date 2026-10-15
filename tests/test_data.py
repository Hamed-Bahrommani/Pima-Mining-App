from __future__ import annotations

import pandas as pd
import pytest

from src.config import ALL_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN
from src.data import (
    DatasetValidationError,
    load_dataset,
    replace_invalid_zeros,
    split_features_target,
    validate_dataset,
)



def sample_frame(rows: int = 12) -> pd.DataFrame:
    data = {
        "Pregnancies": [0, 1, 2, 3, 4, 0] * (rows // 6) + ([0, 1, 2, 3, 4, 0][: rows % 6]),
        "Glucose": [100, 0, 120, 130, 90, 110] * (rows // 6) + ([100, 0, 120, 130, 90, 110][: rows % 6]),
        "BloodPressure": [70, 72, 0, 85, 65, 78] * (rows // 6) + ([70, 72, 0, 85, 65, 78][: rows % 6]),
        "SkinThickness": [20, 22, 0, 30, 18, 25] * (rows // 6) + ([20, 22, 0, 30, 18, 25][: rows % 6]),
        "Insulin": [80, 0, 110, 150, 60, 95] * (rows // 6) + ([80, 0, 110, 150, 60, 95][: rows % 6]),
        "BMI": [25.0, 28.0, 0.0, 33.0, 22.0, 30.0] * (rows // 6) + ([25.0, 28.0, 0.0, 33.0, 22.0, 30.0][: rows % 6]),
        "DiabetesPedigreeFunction": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] * (rows // 6) + ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6][: rows % 6]),
        "Age": [21, 24, 28, 35, 40, 45] * (rows // 6) + ([21, 24, 28, 35, 40, 45][: rows % 6]),
        TARGET_COLUMN: [0, 0, 1, 1, 0, 1] * (rows // 6) + ([0, 0, 1, 1, 0, 1][: rows % 6]),
    }
    return pd.DataFrame(data, columns=ALL_COLUMNS)


def test_schema_validator_accepts_valid_synthetic_data() -> None:
    validate_dataset(sample_frame(), check_reference_counts=False)


def test_schema_validator_rejects_wrong_column_order() -> None:
    frame = sample_frame().loc[:, list(reversed(ALL_COLUMNS))]
    with pytest.raises(DatasetValidationError, match="columns/order"):
        validate_dataset(frame, check_reference_counts=False)


def test_zero_handling_only_targets_five_columns_and_does_not_mutate_input() -> None:
    X = sample_frame().loc[:, FEATURE_COLUMNS]
    original = X.copy(deep=True)
    cleaned = replace_invalid_zeros(X)

    assert pd.isna(cleaned.loc[1, "Glucose"])
    assert pd.isna(cleaned.loc[2, "BloodPressure"])
    assert pd.isna(cleaned.loc[2, "SkinThickness"])
    assert pd.isna(cleaned.loc[1, "Insulin"])
    assert pd.isna(cleaned.loc[2, "BMI"])
    assert cleaned.loc[0, "Pregnancies"] == 0
    pd.testing.assert_frame_equal(X, original)


def test_zero_handling_rejects_target_in_predictors() -> None:
    with pytest.raises(ValueError, match="predictors only"):
        replace_invalid_zeros(sample_frame())


def reference_shape_frame() -> pd.DataFrame:
    rows = 768
    index = pd.Series(range(rows), dtype="int64")
    return pd.DataFrame(
        {
            "Pregnancies": index % 12,
            "Glucose": 70 + (index % 150),
            "BloodPressure": 50 + (index % 55),
            "SkinThickness": 10 + (index % 35),
            "Insulin": 20 + (index % 400),
            "BMI": 18.0 + (index % 30) * 0.5,
            "DiabetesPedigreeFunction": 0.1 + (index % 100) * 0.01,
            "Age": 21 + (index % 60),
            "Outcome": [0] * 500 + [1] * 268,
        },
        columns=ALL_COLUMNS,
    )


def test_reference_schema_and_counts_are_enforced() -> None:
    validate_dataset(reference_shape_frame())
    altered = reference_shape_frame()
    altered.loc[767, "Outcome"] = 0
    with pytest.raises(DatasetValidationError, match="class counts"):
        validate_dataset(altered)


def test_headerless_csv_loader(tmp_path) -> None:
    frame = reference_shape_frame()
    path = tmp_path / "pima.csv"
    frame.to_csv(path, header=False, index=False)
    loaded = load_dataset(path)
    assert loaded.shape == (768, 9)
    assert loaded["Outcome"].value_counts().to_dict() == {0: 500, 1: 268}


def test_validator_rejects_non_binary_target() -> None:
    frame = sample_frame()
    frame.loc[0, TARGET_COLUMN] = 2
    with pytest.raises(DatasetValidationError, match="only 0 and 1"):
        validate_dataset(frame, check_reference_counts=False)


def test_validator_rejects_missing_cells() -> None:
    frame = sample_frame()
    frame.loc[3, "Glucose"] = float("nan")
    with pytest.raises(DatasetValidationError, match="blank/NaN"):
        validate_dataset(frame, check_reference_counts=False)


def test_validator_rejects_non_numeric_columns() -> None:
    frame = sample_frame()
    frame["Glucose"] = frame["Glucose"].astype(str)
    with pytest.raises(DatasetValidationError, match="must be numeric"):
        validate_dataset(frame, check_reference_counts=False)


def test_validator_rejects_wrong_or_missing_column_names() -> None:
    renamed = sample_frame().rename(columns={"Glucose": "Glucose_mg"})
    with pytest.raises(DatasetValidationError, match="columns/order"):
        validate_dataset(renamed, check_reference_counts=False)
    dropped = sample_frame().drop(columns=["Age"])
    with pytest.raises(DatasetValidationError, match="columns/order"):
        validate_dataset(dropped, check_reference_counts=False)


def test_validator_rejects_empty_dataset() -> None:
    with pytest.raises(DatasetValidationError, match="empty"):
        validate_dataset(sample_frame().iloc[0:0], check_reference_counts=False)


def test_reference_check_rejects_wrong_row_count() -> None:
    frame = reference_shape_frame().iloc[:-1]
    with pytest.raises(DatasetValidationError, match="Expected 768 rows"):
        validate_dataset(frame)


def test_split_features_target_separates_outcome() -> None:
    X, y = split_features_target(sample_frame())
    assert list(X.columns) == list(FEATURE_COLUMNS)
    assert y.name == TARGET_COLUMN
    assert set(y.unique()) <= {0, 1}


def test_loader_reports_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="fetch_data.py"):
        load_dataset(tmp_path / "missing.csv")


def test_loader_rejects_csv_with_extra_columns(tmp_path) -> None:
    frame = reference_shape_frame()
    frame.insert(0, "RowId", range(len(frame)))
    path = tmp_path / "wide.csv"
    frame.to_csv(path, header=False, index=False)
    with pytest.raises(DatasetValidationError):
        load_dataset(path)


def test_loader_rejects_csv_with_missing_columns(tmp_path) -> None:
    frame = reference_shape_frame().iloc[:, :-1]
    path = tmp_path / "narrow.csv"
    frame.to_csv(path, header=False, index=False)
    with pytest.raises(DatasetValidationError, match="Expected 9 columns"):
        load_dataset(path)


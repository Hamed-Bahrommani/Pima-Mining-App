"""Dataset loading, schema validation, and stateless zero-value handling."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ALL_COLUMNS,
    DATA_PATH,
    EXPECTED_CLASS_COUNTS,
    EXPECTED_ROWS,
    TARGET_COLUMN,
    ZERO_AS_MISSING_COLUMNS,
)


class DatasetValidationError(ValueError):
    """Raised when the input file does not match the expected dataset schema."""


def validate_dataset(
    frame: pd.DataFrame,
    *,
    check_reference_counts: bool = True,
) -> None:
    """Validate columns, numeric values, target labels, and reference counts.

    Set ``check_reference_counts=False`` for synthetic fixtures or a documented
    alternative version of the dataset; the schema checks still apply.
    """
    if tuple(frame.columns) != ALL_COLUMNS:
        raise DatasetValidationError(
            "Unexpected dataset columns/order. Expected: " + ", ".join(ALL_COLUMNS)
        )

    if frame.empty:
        raise DatasetValidationError("The dataset is empty.")

    non_numeric = [
        column for column in ALL_COLUMNS
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise DatasetValidationError(
            f"These columns must be numeric: {', '.join(non_numeric)}"
        )

    if frame.loc[:, ALL_COLUMNS].isna().any().any():
        raise DatasetValidationError(
            "The raw dataset contains blank/NaN cells; inspect the source file "
            "before applying preprocessing."
        )

    target_values = set(frame[TARGET_COLUMN].unique().tolist())
    if not target_values.issubset({0, 1}):
        raise DatasetValidationError(
            f"{TARGET_COLUMN} must contain only 0 and 1; found {sorted(target_values)}"
        )

    if check_reference_counts:
        if len(frame) != EXPECTED_ROWS:
            raise DatasetValidationError(
                f"Expected {EXPECTED_ROWS} rows, found {len(frame)}. "
                "Check that you downloaded the Pima dataset rather than a different "
                "dataset also named Diabetes."
            )
        counts = frame[TARGET_COLUMN].value_counts().to_dict()
        if counts != EXPECTED_CLASS_COUNTS:
            raise DatasetValidationError(
                "Unexpected class counts. Expected "
                f"{EXPECTED_CLASS_COUNTS}; found {counts}."
            )


def load_dataset(
    path: str | Path = DATA_PATH,
    *,
    check_reference_counts: bool = True,
) -> pd.DataFrame:
    """Load the headerless 9-column CSV and validate it."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Run `python scripts/fetch_data.py` "
            "from the project root, or place the headerless Pima CSV at this path."
        )

    try:
        frame = pd.read_csv(csv_path, header=None)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetValidationError(f"Could not parse dataset CSV: {exc}") from exc

    if frame.shape[1] != len(ALL_COLUMNS):
        raise DatasetValidationError(
            f"Expected {len(ALL_COLUMNS)} columns, found {frame.shape[1]}."
        )
    frame.columns = list(ALL_COLUMNS)

    validate_dataset(frame, check_reference_counts=check_reference_counts)
    return frame


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return a copy of the eight predictors and the binary target."""
    validate_dataset(frame, check_reference_counts=False)
    X = frame.loc[:, frame.columns[:-1]].copy()
    y = frame[TARGET_COLUMN].astype("int64").copy()
    return X, y


def replace_invalid_zeros(X: pd.DataFrame) -> pd.DataFrame:
    """Convert medically implausible zero measurements to NaN, without fitting.

    This function intentionally does not impute. A median imputer must be fit
    on the training partition only, then applied to validation/test data.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("replace_invalid_zeros expects a pandas DataFrame.")
    missing_columns = [col for col in ZERO_AS_MISSING_COLUMNS if col not in X.columns]
    if missing_columns:
        raise ValueError(
            "Missing columns needed for zero handling: " + ", ".join(missing_columns)
        )
    if TARGET_COLUMN in X.columns:
        raise ValueError("Pass predictors only; never preprocess Outcome as a feature.")

    cleaned = X.copy()
    for column in ZERO_AS_MISSING_COLUMNS:
        cleaned[column] = cleaned[column].mask(cleaned[column].eq(0), np.nan)
    return cleaned


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file for reproducibility records."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

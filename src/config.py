"""Shared configuration for the Pima diabetes experiments.

Keep feature definitions and schema constants in one place so the UI,
experiments, and tests cannot silently drift apart.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "pima-indians-diabetes.data.csv"

FEATURE_COLUMNS: tuple[str, ...] = (
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
)
TARGET_COLUMN = "Outcome"
ALL_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS + (TARGET_COLUMN,)

# Widely used interpretation of physiologically impossible zero measurements
# in this dataset. Zero pregnancies is valid and must not be imputed.
ZERO_AS_MISSING_COLUMNS: tuple[str, ...] = (
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
)

F9_COLUMN = "F9_BP_gt_80_and_Glucose_gt_105"
F10_COLUMN = "F10_BP_gt_80"
ENGINEERED_COLUMNS: tuple[str, ...] = (F9_COLUMN, F10_COLUMN)

FEATURE_CONFIGS: dict[str, tuple[str, ...]] = {
    # Six original features: omit SkinThickness and DiabetesPedigreeFunction.
    "6": (
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "Insulin",
        "BMI",
        "Age",
    ),
    "8": FEATURE_COLUMNS,
    "10": FEATURE_COLUMNS + ENGINEERED_COLUMNS,
}

EXPECTED_ROWS = 768
EXPECTED_CLASS_COUNTS = {0: 500, 1: 268}
RANDOM_STATE = 42

# This is the public mirror used by scripts/fetch_data.py. The dataset's
# historical source is the Pima Indians Diabetes dataset distributed via UCI;
# the mirror URL is recorded explicitly rather than presented as a direct
# download from UCI's current (different) dataset page.
DATASET_MIRROR_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)

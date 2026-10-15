# Diabetes Classification Explorer

A two-week Data Mining course project based on Nahzat & Yağanoğlu (2021), *Diabetes Prediction Using Machine Learning Classification Algorithms*.

The goal is to build a reproducible experiment explorer for five classifiers and the paper-inspired six-, eight-, and ten-feature configurations. This is an educational research prototype, **not a clinical diagnostic tool**.

## Project status

**Current milestone: Data validation and preprocessing foundation**

Implemented and tested:

- Central dataset schema and feature-configuration definitions.
- Dataset validation for the expected 768-row, nine-column Pima Indians Diabetes CSV.
- Validation of the expected class distribution: 500 records with `Outcome=0` and 268 with `Outcome=1`.
- Explicit zero-to-missing conversion for `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` only.
- F9/F10 engineered-feature transformer.
- A scikit-learn preprocessing pipeline that fits median imputation, and optional scaling, on training data only.
- Tests for dataset integrity, preprocessing behavior, engineered features, and feature configurations.

The local dataset has been inspected and its SHA-256 checksum recorded in [`data/README.md`](data/README.md). That file also documents the data provenance and observed zero counts.

The project is being developed in stages. The Streamlit interface, model registry, experiment runner, metrics presentation, and published-reference-results display remain subsequent milestones unless their implementation has been verified separately.

## Setup

Python 3.10 or later is recommended.

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the project requirements:

```bash
python -m pip install -r requirements.txt
```

The expected dataset file is:

`data/pima-indians-diabetes.data.csv`

If the dataset is missing, retrieve it using:

```bash
python scripts/fetch_data.py
```

Run the test suite:

```bash
python -m pytest -q
```

See [`data/README.md`](data/README.md) for the dataset source note, validation results, and checksum. See [`reference/methodology_decisions.md`](reference/methodology_decisions.md) for methodological choices and unresolved details in the paper.

## Current methodological choices

- Replace zero with missing only in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`. A value of zero for `Pregnancies` is valid and is preserved.
- Fit the median imputer on training data only, without using target labels to determine imputation values.
- Calculate F9 and F10 after imputation. The paper does not specify whether the thresholds were applied before or after imputation; this is the project's documented choice.
- Use standard scaling for KNN, SVM, and ANN; leave tree-based models unscaled.
- Use a stratified 70/30 train/test split with a fixed seed, with repeated stratified cross-validation planned for stability analysis.
- Keep published paper metrics separate from metrics computed from the project's own predictions.
- Treat the paper's underspecified preprocessing details as limitations, not as established facts. Any sensitivity analysis of label-informed preprocessing must be explicitly identified and kept separate from the primary leakage-safe pipeline.

## Environment

The development environment has reported Python 3.14.4, NumPy 2.5.4, pandas 2.3.3, and scikit-learn 1.9.1. These are observed environment versions, not necessarily pinned dependency versions.

## Scope and limitations

This project is intended for educational experimentation and reproducibility. Results depend on preprocessing choices, data splitting, model configuration, and evaluation methodology. Reproducing a published score does not establish clinical validity or guarantee performance on other populations.

The exact source-file version used by the paper's authors is not specified in the paper. This project uses a documented public mirror of the commonly distributed 768-row Pima CSV; it does not claim that the file is byte-for-byte identical to the authors' original dataset.
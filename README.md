# Diabetes Classification Explorer

A two-week Data Mining course project based on Nahzat & Yağanoğlu (2021), *Diabetes Prediction Using Machine Learning Classification Algorithms*.

The goal is to build a real, reproducible experiment explorer for five classifiers and the paper-inspired six-, eight-, and ten-feature configurations. It is an educational research prototype, **not a clinical diagnostic tool**.

## Current milestone: Day 1 — data and pipeline foundation

This starter includes:

- central dataset schema and feature-configuration definitions;
- schema validation for the expected 768-row, 9-column Pima CSV;
- explicit zero-to-missing conversion for Glucose, BloodPressure, SkinThickness, Insulin, and BMI only;
- the F9/F10 feature transformer;
- a scikit-learn pipeline that fits median imputation (and optional scaling) on training rows only;
- initial unit tests for the data and feature layer.

The dataset itself is not bundled because this environment could not retrieve the external CSV. Download it with the script below on your machine.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/fetch_data.py
python -m pytest -q
```

See `data/README.md` for the data-source note. The app UI, model registry, experiment runner, metrics, and reference-results display have not been implemented yet; those are subsequent milestones.

## Current methodological choices

- Replace zero with missing only in Glucose, BloodPressure, SkinThickness, Insulin, and BMI. Pregnancies=0 remains a valid value.
- Fit the median imputer on training data inside the pipeline.
- Calculate F9/F10 after imputation. The paper does not specify whether these thresholds were applied before or after imputation; this is our documented choice.
- Use standard scaling for KNN, SVM, and ANN; leave tree-based models unscaled.
- Keep published paper metrics separate from metrics computed from our own predictions.

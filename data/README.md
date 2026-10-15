# Dataset acquisition and validation

## Dataset file

The application uses the headerless nine-column Pima Indians Diabetes CSV:

`data/pima-indians-diabetes.data.csv`

Run `python scripts/fetch_data.py` from the project root to download the public mirror configured in `src/config.py`. The download script validates the expected dimensions and records a SHA-256 checksum.

## Source and provenance

The 2021 paper describes the Pima Indians Diabetes Dataset as obtained from UCI. This project uses a public mirror of the standard dataset because the current UCI page with a similar name refers to a different dataset.

The exact source-file version used by the paper's authors is not specified in the paper. Using a mirror therefore does not establish that our file is byte-for-byte identical to the authors' original file.

## Observed validation results

The local CSV was inspected and produced the following results:

- Rows: 768
- Columns: 9
- Outcome = 0: 500 records
- Outcome = 1: 268 records
- Missing CSV cells: 0

These values match the expected dimensions and class distribution.

### Observed zero counts

| Predictor | Zero count |
|---|---:|
| Glucose | 5 |
| BloodPressure | 35 |
| SkinThickness | 227 |
| Insulin | 374 |
| BMI | 11 |

The counts above describe the raw CSV. In the modeling pipeline, zero is treated as a missing measurement for these five predictors only. A value of zero for Pregnancies is valid and is preserved.

## SHA-256 checksum

The observed SHA-256 hash of the local CSV is:

`6BFE5D0F379D17A0E0819B996407E3C09BF80FEBD4287F2ED212190DFFF154AF`

If the file is replaced or downloaded from a different source, recalculate the hash and rerun dataset validation. Do not assume that a replacement file is identical merely because it has the same filename or dimensions.

## Preprocessing notes

- Keep the raw CSV unchanged.
- Convert zero to missing only in Glucose, BloodPressure, SkinThickness, Insulin, and BMI.
- Fit median imputation on training data only, without using target labels to determine imputation values.
- Calculate F9 and F10 after imputation in the primary pipeline. The paper does not specify the exact order.
- Treat these rules as project implementation choices where the paper leaves details unspecified.

## Loader validation

`src.data.load_dataset()` validates the expected column schema, 768-row count, and class distribution (500 records with Outcome=0 and 268 with Outcome=1).


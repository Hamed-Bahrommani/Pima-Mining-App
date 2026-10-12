# Dataset acquisition

The application expects the headerless nine-column Pima Indians Diabetes CSV at:

`data/pima-indians-diabetes.data.csv`

Run `python scripts/fetch_data.py` from the project root to download the public mirror configured in `src/config.py`. The script checks the row count and CSV width and records a SHA-256 checksum next to the dataset.

**Source note:** the paper describes the Pima Indians Diabetes Dataset as obtained from UCI. The current UCI page with a similarly named “Diabetes” dataset describes a different, longitudinal dataset. This project therefore uses a public mirror of the commonly distributed 768-row Pima CSV and documents the mirror URL explicitly rather than representing it as a direct download from UCI's current page.

Before accepting a downloaded file, `src.data.load_dataset()` checks the expected column schema, 768 rows, and class counts (500 Outcome=0; 268 Outcome=1).

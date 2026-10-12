"""Download the public Pima CSV mirror and record its SHA-256 checksum.

Run from the project root: python scripts/fetch_data.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import urllib.error
import urllib.request
import sys
from pathlib import Path

# Ensure the project root is importable when invoked as `python scripts/fetch_data.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PATH, DATASET_MIRROR_URL, ALL_COLUMNS, EXPECTED_ROWS


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        DATASET_MIRROR_URL,
        headers={"User-Agent": "diabetes-mining-course-project/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(
            "Could not download the dataset. Check your internet connection or "
            "download the headerless CSV manually from:\n"
            f"{DATASET_MIRROR_URL}\n"
            f"Save it as: {DATA_PATH}\nError: {exc}"
        ) from exc

    # Validate the contents before writing, so a changed or wrong download
    # cannot silently become the local project dataset.
    try:
        text = payload.decode("utf-8-sig")
        rows = [row for row in csv.reader(io.StringIO(text)) if row]
        if len(rows) != EXPECTED_ROWS:
            raise SystemExit(f"Expected {EXPECTED_ROWS} CSV rows; downloaded {len(rows)}.")
        wrong_width = [index + 1 for index, row in enumerate(rows) if len(row) != len(ALL_COLUMNS)]
        if wrong_width:
            raise SystemExit(
                f"Some CSV rows do not have {len(ALL_COLUMNS)} fields; first bad rows: "
                f"{wrong_width[:5]}"
            )
        for row_number, row in enumerate(rows, start=1):
            for value in row:
                float(value)
        outcomes = [int(float(row[-1])) for row in rows]
        class_counts = {label: outcomes.count(label) for label in (0, 1)}
        if set(outcomes) != {0, 1} or class_counts != {0: 500, 1: 268}:
            raise SystemExit(
                "Unexpected Outcome class counts. Expected {0: 500, 1: 268}; "
                f"found {class_counts}."
            )
    except (UnicodeDecodeError, ValueError, csv.Error) as exc:
        raise SystemExit(f"Downloaded file failed numeric CSV validation: {exc}") from exc

    DATA_PATH.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    checksum_path = DATA_PATH.with_suffix(DATA_PATH.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {DATA_PATH.name}\n", encoding="utf-8")
    print(f"Saved {DATA_PATH}")
    print(f"Rows: {len(rows)}")
    print(f"SHA-256: {digest}")
    print(f"Checksum saved to {checksum_path}")
    print("Next: python -m pytest")


if __name__ == "__main__":
    main()

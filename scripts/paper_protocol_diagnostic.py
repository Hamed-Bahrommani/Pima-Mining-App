"""Diagnostic: how much can label-informed imputation change Random Forest accuracy?

This is NOT part of the app pipeline and NOT a reproduction claim. It compares,
one change at a time, how the way zero measurements are replaced affects a
default-parameter Random Forest (threshold 0.5, accuracy as the headline metric)
on repeated stratified 70/30 splits.

Variants
  clean               our app pipeline: train-only, label-free median imputation
  train_class_median  training rows imputed with their own class median
                      (train rows only); test rows imputed label-free
  leaky_median_5      class medians computed on ALL rows, applied to ALL rows
                      before splitting (uses test labels: leakage) - 5 columns
  leaky_mean_5        same, class means
  leaky_median_8      same as leaky_median_5 but treating zero as missing in
                      every predictor column (a literal reading of the paper)

The leaky variants are a reading of the paper's wording, which does not fully
specify its procedure. They exist to measure leakage, not to estimate how
a deployed model would perform.

Run from the project root:
    python -m scripts.paper_protocol_diagnostic --seeds 100 --out diagnostic_results.csv
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH,
    FEATURE_COLUMNS,
    FEATURE_CONFIGS,
    ZERO_AS_MISSING_COLUMNS,
)
from src.data import load_dataset, split_features_target
from src.features import EngineeredFeaturesTransformer
from src.pipeline import build_pipeline

TEST_SIZE = 0.30
PAPER_RF_ACCURACY = 0.8831
PAPER_CONFUSION = np.array([[139, 11], [16, 65]])  # Figure 4; rows = actual


def score(y_true, y_pred):
    macro = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    binary = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": macro[0],
        "macro_recall": macro[1],
        "macro_f1": macro[2],
        "class1_precision": binary[0],
        "class1_recall": binary[1],
        "class1_f1": binary[2],
    }


def class_conditional_impute(X, y, columns, statistic):
    """Replace zeros with the per-class mean/median of the non-zero values.

    Uses the labels in y. Reading of the paper's description, not a faithful copy.
    """
    imputed = X.astype(float).copy()
    labels = y.to_numpy()
    for column in columns:
        values = imputed[column].to_numpy(copy=True)
        for label in (0, 1):
            in_class = labels == label
            valid = in_class & (values != 0)
            fill = getattr(pd.Series(values[valid]), statistic)()
            values[in_class & (values == 0)] = fill
        imputed[column] = values
    return imputed


def add_features(X_imputed, feature_config):
    """F9/F10 from the imputed values, then select the requested configuration."""
    engineered = EngineeredFeaturesTransformer().fit_transform(X_imputed)
    return engineered.loc[:, list(FEATURE_CONFIGS[feature_config])]


def fit_predict(X_train, y_train, X_test, seed):
    model = RandomForestClassifier(random_state=seed)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def run_clean(X, y, train_idx, test_idx, feature_config, seed):
    pipeline = build_pipeline(
        RandomForestClassifier(random_state=seed),
        feature_config=feature_config,
        scale_features=False,
    )
    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
    return pipeline.predict(X.iloc[test_idx])


def run_train_class_median(X, y, train_idx, test_idx, feature_config, seed):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test = X.iloc[test_idx].astype(float).copy()
    train_imputed = class_conditional_impute(
        X_train, y_train, ZERO_AS_MISSING_COLUMNS, "median"
    )
    for column in ZERO_AS_MISSING_COLUMNS:
        train_values = X_train[column].to_numpy()
        overall_median = float(np.median(train_values[train_values != 0]))
        X_test[column] = X_test[column].where(X_test[column] != 0, overall_median)
    return fit_predict(
        add_features(train_imputed, feature_config),
        y_train,
        add_features(X_test, feature_config),
        seed,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--config", default="10", choices=sorted(FEATURE_CONFIGS))
    parser.add_argument("--out", default=None, help="optional CSV of per-seed results")
    args = parser.parse_args()

    frame = load_dataset(args.data)
    X, y = split_features_target(frame)
    config = args.config
    print(f"Random Forest (sklearn defaults), {config} features, "
          f"{args.seeds} stratified {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} splits")

    # Paper reference: what the printed confusion matrix implies.
    tn, fp, fn, tp = PAPER_CONFUSION.ravel()
    y_true = np.array([0] * (tn + fp) + [1] * (fn + tp))
    y_pred = np.array([0] * tn + [1] * fp + [0] * fn + [1] * tp)
    paper = score(y_true, y_pred)
    print("Paper Figure 4 matrix (n=%d): accuracy %.4f, macro P/R/F1 %.3f/%.3f/%.3f"
          % (PAPER_CONFUSION.sum(), paper["accuracy"], paper["macro_precision"],
             paper["macro_recall"], paper["macro_f1"]))

    # Label-informed imputations are computed once on all rows (that is the leak).
    leaky = {
        "leaky_median_5": class_conditional_impute(X, y, ZERO_AS_MISSING_COLUMNS, "median"),
        "leaky_mean_5": class_conditional_impute(X, y, ZERO_AS_MISSING_COLUMNS, "mean"),
        "leaky_median_8": class_conditional_impute(X, y, FEATURE_COLUMNS, "median"),
    }
    leaky_features = {name: add_features(imputed, config) for name, imputed in leaky.items()}

    rows = []
    started = time.time()
    for seed in range(args.seeds):
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=TEST_SIZE, stratify=y, random_state=seed
        )
        y_test = y.iloc[test_idx]
        predictions = {
            "clean": run_clean(X, y, train_idx, test_idx, config, seed),
            "train_class_median": run_train_class_median(
                X, y, train_idx, test_idx, config, seed
            ),
        }
        for name, features in leaky_features.items():
            predictions[name] = fit_predict(
                features.iloc[train_idx], y.iloc[train_idx], features.iloc[test_idx], seed
            )
        # Rows that had at least one zero (= an imputed value) in the 5 columns.
        had_zero = (
            X.iloc[test_idx][list(ZERO_AS_MISSING_COLUMNS)] == 0
        ).any(axis=1).to_numpy()
        truth = y_test.to_numpy()
        for name, predicted in predictions.items():
            correct = predicted == truth
            rows.append({
                "variant": name,
                "seed": seed,
                **score(y_test, predicted),
                "acc_rows_with_zero": correct[had_zero].mean(),
                "acc_rows_without_zero": correct[~had_zero].mean(),
                "share_rows_with_zero": had_zero.mean(),
            })
    print(f"finished in {time.time() - started:.0f}s\n")

    results = pd.DataFrame(rows)
    order = ["clean", "train_class_median", "leaky_median_5", "leaky_mean_5", "leaky_median_8"]

    summary = results.groupby("variant")["accuracy"].agg(
        mean="mean",
        sd="std",
        p2_5=lambda s: s.quantile(0.025),
        p97_5=lambda s: s.quantile(0.975),
        max="max",
    )
    summary["share_ge_88.31"] = results.groupby("variant")["accuracy"].apply(
        lambda s: float((s >= PAPER_RF_ACCURACY - 1e-9).mean())
    )
    for column in ("macro_precision", "macro_recall", "macro_f1", "class1_f1"):
        summary[column] = results.groupby("variant")[column].mean()
    print("Test accuracy across seeds (and mean of other metrics)")
    print(summary.loc[order].to_string(float_format=lambda v: f"{v:.4f}"))

    accuracy = results.pivot(index="seed", columns="variant", values="accuracy")
    print("\nPaired difference vs clean (same seeds), accuracy")
    paired = pd.DataFrame({
        "mean_diff": [(accuracy[name] - accuracy["clean"]).mean() for name in order[1:]],
        "sd_diff": [(accuracy[name] - accuracy["clean"]).std() for name in order[1:]],
    }, index=order[1:])
    print(paired.to_string(float_format=lambda v: f"{v:+.4f}"))

    print("\nWhere does the gain come from? Mean test accuracy by row type")
    print("(rows with a zero = at least one of the 5 columns was imputed)")
    by_row_type = results.groupby("variant")[
        ["acc_rows_with_zero", "acc_rows_without_zero", "share_rows_with_zero"]
    ].mean()
    print(by_row_type.loc[order].to_string(float_format=lambda v: f"{v:.4f}"))

    if args.out:
        results.to_csv(args.out, index=False)
        print(f"\nper-seed results written to {args.out}")


main()

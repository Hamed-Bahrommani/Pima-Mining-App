"""Interactive diabetes classification experiment explorer."""

import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import FEATURE_COLUMNS, TARGET_COLUMN, ZERO_AS_MISSING_COLUMNS
from src.data import load_dataset
from src.experiments import compare_experiments

STYLES_PATH = Path(__file__).resolve().parent / "assets" / "styles.css"

ACCENT = "#83A9E6"
MODEL_LABELS = {
    "knn": "K-Nearest Neighbors",
    "random_forest": "Random Forest",
    "svm": "Support Vector Machine",
    "ann": "Artificial Neural Network",
    "decision_tree": "Decision Tree",
}
FEATURE_CONFIG_LABELS = {
    "6": "6 features",
    "8": "8 original features",
    "10": "10 features (including F9/F10)",
}
# Chart series: ordered by feature count, one muted blue ramp.
CONFIG_ORDER = ["6 features", "8 features", "10 features"]
CONFIG_COLORS = {
    "6 features": "#5F7FB0",
    "8 features": "#83A9E6",
    "10 features": "#C5D6F2",
}


def load_styles() -> None:
    if STYLES_PATH.exists():
        css = STYLES_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def summarize_warnings(caught) -> list[str]:
    """Reduce recorded warnings to unique one-line entries (shown, not hidden)."""
    messages: list[str] = []
    for item in caught:
        first_line = (str(item.message).strip().splitlines() or [""])[0]
        entry = f"{item.category.__name__}: {first_line}"
        if entry not in messages:
            messages.append(entry)
    return messages


def accuracy_chart(results: pd.DataFrame, value_column: str) -> None:
    chart = results.pivot(
        index="model_label",
        columns="configuration",
        values=value_column,
    )
    chart = chart.reindex(
        index=[name for name in MODEL_LABELS.values() if name in chart.index],
        columns=[name for name in CONFIG_ORDER if name in chart.columns],
    )
    st.bar_chart(
        chart,
        color=[CONFIG_COLORS[name] for name in chart.columns],
        stack=False,
        horizontal=True,
    )


st.set_page_config(
    page_title="Pima Diabetes Experiment Explorer",
    layout="wide",
)
load_styles()

st.markdown(
    '<p class="app-eyebrow">Data Mining · Educational research prototype</p>'
    '<h1 class="app-title">Pima Diabetes Experiment Explorer</h1>'
    '<p class="app-subtitle">Compare five classifiers across three feature '
    "configurations on the Pima Indians Diabetes dataset, following "
    "Nahzat &amp; Yağanoğlu (2021). This is an educational tool, "
    "not a diagnostic system.</p>",
    unsafe_allow_html=True,
)


@st.cache_data
def get_dataset():
    return load_dataset()


try:
    dataset = get_dataset()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Experiment settings")
st.sidebar.caption("Choose what to compare, then run the comparison.")

FEATURE_PILLS = {"6": "6 features", "8": "8 features", "10": "10 + F9/F10"}
MODEL_PILLS = {
    "knn": "KNN",
    "random_forest": "Random Forest",
    "svm": "SVM",
    "ann": "ANN",
    "decision_tree": "Decision Tree",
}

picked_configs = st.sidebar.pills(
    "Feature configurations",
    options=list(FEATURE_PILLS),
    selection_mode="multi",
    default=list(FEATURE_PILLS),
    format_func=lambda value: FEATURE_PILLS[value],
)
picked_models = st.sidebar.pills(
    "Classifiers",
    options=list(MODEL_PILLS),
    selection_mode="multi",
    default=list(MODEL_PILLS),
    format_func=lambda value: MODEL_PILLS[value],
)
# Keep a stable order regardless of click order.
feature_configs = [key for key in FEATURE_PILLS if key in (picked_configs or [])]
selected_models = [key for key in MODEL_PILLS if key in (picked_models or [])]

folds = st.sidebar.slider(
    "Cross-validation folds",
    min_value=3,
    max_value=10,
    value=5,
)

can_run = bool(feature_configs and selected_models)
run_button = st.sidebar.button(
    "Run comparison",
    type="primary",
    disabled=not can_run,
)
if not can_run:
    st.sidebar.caption("Select at least one feature configuration and one classifier.")

# --------------------------------------------------------- dataset overview
st.header("Dataset overview")

total_rows = len(dataset)
positive_count = int((dataset[TARGET_COLUMN] == 1).sum())
negative_count = int((dataset[TARGET_COLUMN] == 0).sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Records", f"{total_rows:,}")
col2.metric("Original predictors", len(FEATURE_COLUMNS))
col3.metric("Positive outcomes", positive_count)
col4.metric("Negative outcomes", negative_count)

overview_tab, missing_tab, data_tab = st.tabs(
    ["Outcome distribution", "Zero-value review", "Dataset preview"]
)

with overview_tab:
    outcome_counts = (
        dataset[TARGET_COLUMN]
        .value_counts()
        .reindex([0, 1], fill_value=0)
        .rename(index={0: "Outcome 0", 1: "Outcome 1"})
    )
    st.bar_chart(outcome_counts, color=ACCENT)
    st.caption(
        "Outcome 0 and Outcome 1 are the dataset's class labels, "
        "not individual medical diagnoses."
    )

with missing_tab:
    zero_counts = pd.DataFrame({
        "Predictor": list(ZERO_AS_MISSING_COLUMNS),
        "Zero values": [
            int((dataset[column] == 0).sum())
            for column in ZERO_AS_MISSING_COLUMNS
        ],
    })
    zero_counts["Share of records (%)"] = (
        100 * zero_counts["Zero values"] / total_rows
    ).round(1)
    st.write(
        "In these predictors, zero values are treated as missing "
        "by the experiment preprocessing pipeline."
    )
    st.dataframe(zero_counts, hide_index=True, width="stretch")
    st.caption("Zero Pregnancies is a valid value and is left unchanged.")

with data_tab:
    st.dataframe(dataset.head(20), width="stretch")
    st.caption(f"Showing 20 of {total_rows} records.")


# ------------------------------------------------------ experiment explorer
st.header("Experiment explorer")

st.caption(
    "Compare classifiers across feature configurations using stratified "
    "cross-validation on the training partition and a shared held-out test set."
)


st.subheader("Configuration")
if can_run:
    st.write(
        f"{len(selected_models)} classifier(s) × {len(feature_configs)} "
        f"feature configuration(s), {folds}-fold cross-validation, "
        "plus a held-out test split."
    )
else:
    st.warning("Select at least one feature configuration and one classifier.")


if run_button:
    X = dataset.drop(columns=[TARGET_COLUMN])
    y = dataset[TARGET_COLUMN]

    try:
        with st.spinner(
            "Comparing classifiers and optimizing selected models..."
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")

                results = compare_experiments(
                    X,
                    y,
                    feature_configs=feature_configs,
                    model_names=selected_models,
                    test_size=0.30,
                    folds=folds,
                )

        st.session_state["experiment_results"] = results
        st.session_state["experiment_folds"] = folds
        st.session_state["experiment_warnings"] = summarize_warnings(caught)

    except (ValueError, RuntimeError) as exc:
        st.error(f"Experiment failed: {exc}")

st.subheader("Results")

if "experiment_results" in st.session_state:
    results = st.session_state["experiment_results"].copy()

    results["model_label"] = results["model"].map(MODEL_LABELS)
    results["configuration"] = results["feature_config"].map(
        lambda value: f"{value} features"
    )

    st.caption(
        f"{results['model'].nunique()} classifier(s) × "
        f"{results['feature_config'].nunique()} feature configuration(s), "
        f"{st.session_state.get('experiment_folds', '?')}-fold cross-validation. "
        "Settings above may have changed since this run."
    )

    display_columns = [
        "model_label",
        "feature_config",
        "accuracy_mean",
        "accuracy_std",
        "f1_positive_mean",
        "f1_positive_std",
        "test_accuracy",
        "test_precision_positive",
        "test_recall_positive",
        "test_f1_positive",
    ]

    table = results[display_columns].rename(columns={
        "model_label": "Classifier",
        "feature_config": "Features",
        "accuracy_mean": "CV accuracy",
        "accuracy_std": "CV acc. SD",
        "f1_positive_mean": "CV F1",
        "f1_positive_std": "CV F1 SD",
        "test_accuracy": "Test accuracy",
        "test_precision_positive": "Test precision",
        "test_recall_positive": "Test recall",
        "test_f1_positive": "Test F1",
    })
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=35 * (len(table) + 1) + 3,
        column_config={
            "Classifier": st.column_config.TextColumn(width="medium"),
            "Features": st.column_config.TextColumn(width="small"),
            **{
                name: st.column_config.NumberColumn(format="%.3f", width="small")
                for name in table.columns[2:]
            },
        },
    )
    
    st.caption(
        "CV accuracy and CV F1 are means across training-partition folds; "
        "their SD columns show fold-to-fold variability. Test metrics are "
        "calculated on the shared held-out test partition. Precision, recall, "
        "and F1 use class 1 (Outcome 1) as the positive class. CV results "
        "guide model comparison; the held-out test results provide a separate "
        "evaluation."
    )

    
    st.markdown(
        '<p class="notice">Cross-validation uses only the training partition. '
        "All classifiers and feature configurations share the same held-out "
        "test partition, which is reserved for final evaluation. Results are "
        "exploratory comparisons; small differences between models should not "
        "be overinterpreted.</p>",
        unsafe_allow_html=True,
    )

    run_warnings = st.session_state.get("experiment_warnings", [])
    if run_warnings:
        with st.expander(f"Warnings raised during this run ({len(run_warnings)})"):
            st.caption(
                "These come from the underlying libraries, for example a neural "
                "network that reached its iteration limit. They are shown, "
                "not suppressed."
            )
            for entry in run_warnings:
                st.code(entry, language=None)

    left, right = st.columns(2)
    with left:
        st.subheader("Cross-validation accuracy")
        accuracy_chart(results, "accuracy_mean")
    with right:
        st.subheader("Held-out test accuracy")
        accuracy_chart(results, "test_accuracy")

else:
    st.info("No results yet. Choose your settings in the sidebar, then click **Run comparison**.")

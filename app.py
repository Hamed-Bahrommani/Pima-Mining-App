"""Interactive diabetes classification experiment explorer."""

import streamlit as st
import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_COLUMN, ZERO_AS_MISSING_COLUMNS
from src.data import load_dataset
from src.experiments import compare_experiments


st.set_page_config(
    page_title="Pima Diabetes Experiment Explorer",
    page_icon="🧪",
    layout="wide",
)

st.title("Pima Diabetes Experiment Explorer")
st.caption(
    "Explore dataset characteristics and compare machine-learning classifiers. "
    "This is an educational research tool, not a diagnostic system."
)


@st.cache_data
def get_dataset():
    return load_dataset()


try:
    dataset = get_dataset()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()


# Sidebar controls
st.sidebar.header("Experiment settings")

feature_configs = st.sidebar.multiselect(
    "Feature configurations",
    options=["6", "8", "10"],
    default=["6", "8", "10"],
    format_func=lambda value: {
        "6": "6 features",
        "8": "8 original features",
        "10": "10 features (including F9/F10)",
    }[value],
)

model_labels = {
    "knn": "K-Nearest Neighbors",
    "random_forest": "Random Forest",
    "svm": "Support Vector Machine",
    "ann": "Artificial Neural Network",
    "decision_tree": "Decision Tree",
}

selected_models = st.sidebar.multiselect(
    "Classifiers",
    options=list(model_labels),
    default=list(model_labels),
    format_func=lambda value: model_labels[value],
)

folds = st.sidebar.slider(
    "Cross-validation folds",
    min_value=3,
    max_value=10,
    value=5,
)

run_button = st.sidebar.button(
    "Run comparison",
    type="primary",
    disabled=not feature_configs or not selected_models,
)


# Dataset overview
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
    st.bar_chart(outcome_counts)
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
    st.write(
        "In selected predictors, zero values are treated as missing "
        "by the experiment preprocessing pipeline."
    )
    st.dataframe(zero_counts, hide_index=True, use_container_width=True)

with data_tab:
    st.dataframe(dataset.head(20), use_container_width=True)
    st.caption(f"Showing 20 of {total_rows} records.")


# Experiment explorer
st.header("Experiment explorer")
st.write(
    "Compare classifiers across feature configurations using stratified "
    "cross-validation and a held-out test split."
)

if run_button:
    X = dataset.drop(columns=[TARGET_COLUMN])
    y = dataset[TARGET_COLUMN]

    with st.spinner("Running experiments… this may take a little while."):
        try:
            results = compare_experiments(
                X,
                y,
                feature_configs=tuple(feature_configs),
                model_names=tuple(selected_models),
                folds=folds,
            )
        except (ValueError, RuntimeError) as exc:
            st.error(f"The comparison could not be completed: {exc}")
            st.stop()

    st.session_state["experiment_results"] = results

if "experiment_results" in st.session_state:
    results = st.session_state["experiment_results"].copy()

    results["model_label"] = results["model"].map(model_labels)
    results["configuration"] = results["feature_config"].map(
        lambda value: f"{value} features"
    )

    st.subheader("Results table")

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

    st.dataframe(
        results[display_columns].rename(columns={
            "model_label": "Classifier",
            "feature_config": "Features",
            "accuracy_mean": "CV accuracy mean",
            "accuracy_std": "CV accuracy std",
            "f1_positive_mean": "CV F1 mean",
            "f1_positive_std": "CV F1 std",
            "test_accuracy": "Test accuracy",
            "test_precision_positive": "Test precision",
            "test_recall_positive": "Test recall",
            "test_f1_positive": "Test F1",
        }).round(4),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Cross-validation accuracy")
    accuracy_chart = results.pivot(
        index="model_label",
        columns="configuration",
        values="accuracy_mean",
    )
    st.bar_chart(accuracy_chart)

    st.subheader("Held-out test accuracy")
    test_chart = results.pivot(
        index="model_label",
        columns="configuration",
        values="test_accuracy",
    )
    st.bar_chart(test_chart)

else:
    st.info("Choose your settings in the sidebar, then click **Run comparison**.")



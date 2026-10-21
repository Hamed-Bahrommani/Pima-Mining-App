"""Interactive diabetes classification experiment explorer."""

import warnings
from pathlib import Path

import plotly.express as px

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
    chart_labels = {
        "K-Nearest Neighbors": "KNN",
        "Support Vector Machine": "SVM",
        "Artificial Neural Network": "ANN",
    }

    chart = results.pivot(
        index="model_label",
        columns="configuration",
        values=value_column,
    )

    chart = chart.rename(index=chart_labels)

    chart = chart.reindex(
        index=[
            chart_labels.get(name, name)
            for name in MODEL_LABELS.values()
            if chart_labels.get(name, name) in chart.index
        ],
        columns=[
            name for name in CONFIG_ORDER
            if name in chart.columns
        ],
    )

    # Plotly expects one row per classifier/configuration pair.
    plot_data = (
        chart.rename_axis("Classifier")
        .reset_index()
        .melt(
            id_vars="Classifier",
            var_name="Feature configuration",
            value_name="Score",
        )
        .dropna(subset=["Score"])
    )

    if plot_data.empty:
        st.info("No chart data available for this comparison.")
        return

    fig = px.bar(
        plot_data,
        x="Score",
        y="Classifier",
        color="Feature configuration",
        barmode="group",
        orientation="h",
        category_orders={
            "Classifier": [
                chart_labels.get(name, name)
                for name in MODEL_LABELS.values()
                if chart_labels.get(name, name)
                in plot_data["Classifier"].unique()
            ][::-1],
            "Feature configuration": CONFIG_ORDER,
        },
        color_discrete_map=CONFIG_COLORS,
        custom_data=["Feature configuration"],
        labels={
            "Score": "Score",
            "Classifier": "",
            "Feature configuration": "Features",
        },
        hover_data={
            "Score": ":.3f",
            "Feature configuration": True,
            "Classifier": True,
        },
    )

    fig.update_traces(
        marker_line_width=0,
        opacity=0.95,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "%{customdata[0]}<br>"
            "Score: %{x:.3f}"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Manrope, Estedad, Segoe UI, sans-serif",
            color="#C4C9CE",
            size=12,
        ),
        xaxis=dict(
            title=None,
            range=[0, 1],
            tickformat=".0%",
            gridcolor="#363C44",
            zeroline=False,
        ),
        yaxis=dict(
            title=None,
            automargin=True,
            showgrid=False,
            tickfont=dict(size=12),
        ),
        legend=dict(
            title=None,
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        margin=dict(l=8, r=12, t=36, b=12),
        height=max(300, 76 * len(chart.index)),
        bargap=0.34,
        bargroupgap=0.14,
        hoverlabel=dict(
            bgcolor="#252A30",
            bordercolor="#46515E",
            font_color="#F1F3F5",
        ),
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False},
    )



st.set_page_config(
    page_title="Pima Diabetes Experiment Explorer",
    layout="wide",
)
load_styles()


st.markdown(
    '<div class="app-header">'
    '<p class="app-eyebrow"><span class="eyebrow-dot"></span>'
    'DATA MINING <span class="eyebrow-separator">/</span> '
    'EDUCATIONAL RESEARCH PROTOTYPE</p>'
    '<h1 class="app-title">Pima Diabetes Experiment Explorer</h1>'
    '<p class="app-subtitle">Compare five classifiers across three feature '
    'configurations on the Pima Indians Diabetes dataset, following '
    'Nahzat &amp; Yağanoğlu (2021). This is an educational tool, '
    'not a diagnostic system.</p>'
    '</div>',
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


metric_cols = st.columns(4, gap="medium")

metrics = [
    ("RECORDS", f"{total_rows:,}", "Dataset size"),
    ("PREDICTORS", str(len(FEATURE_COLUMNS)), "Original input variables"),
    ("POSITIVE OUTCOMES", f"{positive_count:,}", "Outcome = 1"),
    ("NEGATIVE OUTCOMES", f"{negative_count:,}", "Outcome = 0"),
]

for col, (label, value, description) in zip(metric_cols, metrics):
    with col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-card-label">{label}</div>'
            f'<div class="metric-card-value">{value}</div>'
            f'<div class="metric-card-note">{description}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


overview_tab, missing_tab, data_tab = st.tabs(
    ["Outcome distribution", "Zero-value review", "Dataset preview"]
)

with overview_tab:
    outcome_counts = (
        dataset[TARGET_COLUMN]
        .value_counts()
        .reindex([0, 1], fill_value=0)
        .rename(index={0: "Outcome 0 · Negative", 1: "Outcome 1 · Positive"})
    )

    outcome_plot = outcome_counts.rename_axis("Outcome").reset_index()
    outcome_plot.columns = ["Outcome", "Records"]
    outcome_plot["Outcome"] = outcome_plot["Outcome"].replace({
        "Outcome 0": "Outcome 0 · Negative",
        "Outcome 1": "Outcome 1 · Positive",
    })

    fig = px.bar(
        outcome_plot,
        x="Records",
        y="Outcome",
        orientation="h",
        text="Records",
        color_discrete_sequence=[ACCENT],
        labels={"Records": "Number of records", "Outcome": ""},
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Records: %{x}<extra></extra>",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(title=None, gridcolor="#363C44", rangemode="tozero"),
        yaxis=dict(title=None, showgrid=False, automargin=True),
        margin=dict(l=12, r=45, t=12, b=12),
        height=220,
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False},
    )

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
            "Comparing classifiers and optimizing selected models. This may take a while..."
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
        "accuracy_std": "CV SD",
        "f1_positive_mean": "CV F1",
        "f1_positive_std": "F1 SD",
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
            "Classifier": st.column_config.TextColumn(
                width="medium",
                help="Classification algorithm used.",
            ),
            "Features": st.column_config.TextColumn(
                width="small",
                help="Number of features used in the experiment.",
            ),
            "CV accuracy": st.column_config.NumberColumn(
                format="%.3f",
                width="small",
                help="Mean accuracy across training-partition CV folds.",
            ),
            "CV SD": st.column_config.NumberColumn(
                format="%.3f",
                width="small",
                help="Standard deviation of accuracy across CV folds.",
            ),
            "CV F1": st.column_config.NumberColumn(
                format="%.3f",
                width="small",
                help="Mean positive-class F1 across CV folds.",
            ),
            "F1 SD": st.column_config.NumberColumn(
                format="%.3f",
                width="small",
                help="Standard deviation of positive-class F1 across CV folds.",
            ),
            "Test accuracy": st.column_config.NumberColumn(
                format="%.3f",
                width="small",
                help="Accuracy on the held-out test partition.",
            ),
            "Test precision": st.column_config.NumberColumn(
                format="%.3f",
                width="small",
                help="Positive-class precision on the held-out test partition.",
            ),
            "Test recall": st.column_config.NumberColumn(
                format="%.3f",
                width="small",
                help="Positive-class recall on the held-out test partition.",
            ),
            "Test F1": st.column_config.NumberColumn(
                format="%.3f",
                width="small",
                help="Positive-class F1 on the held-out test partition.",
            ),
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

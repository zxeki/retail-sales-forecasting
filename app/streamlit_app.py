"""Retail Sales Forecasting dashboard.

Interactive Streamlit app to explore historical sales trends and inspect the
LightGBM forecast against the seasonal-naive baseline on the validation window.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from retail_sales_forecasting.dashboard_data import (
    daily_totals,
    family_summary,
    filter_options,
    filter_predictions,
    metric_comparison,
    slice_metrics,
    store_summary,
)
from retail_sales_forecasting.data_loader import (
    available_datasets,
    load_available_datasets,
    load_dataset,
)
from retail_sales_forecasting.forecast import (
    build_forecast_run,
    has_forecast_artifact,
    load_forecast_run,
    save_forecast_run,
)

st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)


@st.cache_data(show_spinner="Loading datasets...")
def load_historical() -> dict:
    """Load the raw datasets used for historical views."""
    return load_available_datasets()


@st.cache_data(show_spinner="Running the forecasting pipeline...")
def load_forecast_cached() -> pd.DataFrame:
    """Load or build the forecast run and return a serializable bundle."""
    run = load_forecast_run()
    metrics = run.metrics.copy()
    metrics.insert(0, "cutoff", run.cutoff.strftime("%Y-%m-%d"))
    return run.predictions, metrics


def render_header(datasets: dict) -> None:
    st.title("Retail Sales Forecasting")
    st.caption(
        "Historical sales trends and a leakage-safe LightGBM forecast compared "
        "against a seasonal-naive baseline on a 16-day validation window."
    )


def render_filters(predictions: pd.DataFrame) -> tuple:
    options = filter_options(predictions)
    st.sidebar.header("Filters")

    families = st.sidebar.multiselect(
        "Product family",
        options=options["families"],
        default=options["families"],
    )
    stores = st.sidebar.multiselect(
        "Store number",
        options=options["stores"],
        default=options["stores"],
    )
    date_range = st.sidebar.date_input(
        "Validation date range",
        value=(options["date_min"].date(), options["date_max"].date()),
        min_value=options["date_min"].date(),
        max_value=options["date_max"].date(),
    )
    return families, stores, date_range


def render_kpis(filtered: pd.DataFrame, metrics: pd.DataFrame) -> None:
    st.subheader("Validation performance")
    model_row = metrics.iloc[0]
    baseline_row = metrics.iloc[1] if len(metrics) > 1 else None

    slice_stats = slice_metrics(filtered)
    columns = st.columns(4)
    columns[0].metric("RMSLE (LightGBM)", f"{model_row['rmsle']:.4f}")
    if baseline_row is not None:
        delta = (model_row["rmsle"] - baseline_row["rmsle"]) / baseline_row["rmsle"] * 100
        columns[1].metric(
            "RMSLE (baseline)",
            f"{baseline_row['rmsle']:.4f}",
            delta=f"{delta:.1f}%",
            delta_color="inverse",
        )
    columns[2].metric("MAE (selected slice)", f"{slice_stats['mae']:.2f}")
    columns[3].metric("Rows (selected slice)", f"{slice_stats['rows']:,}")


def render_trends(datasets: dict) -> None:
    st.subheader("Historical sales trends")
    if "train" not in datasets:
        st.info("Historical training data is not available.")
        return

    train = datasets["train"]
    daily = (
        train.groupby("date", as_index=False)["sales"]
        .sum()
        .sort_values("date")
    )
    monthly = (
        train.assign(month=train["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)["sales"]
        .sum()
    )

    left, right = st.columns(2)
    with left:
        fig = px.line(monthly, x="month", y="sales", title="Monthly total sales")
        st.plotly_chart(fig, width='stretch')
    with right:
        family_totals = (
            train.groupby("family", as_index=False)["sales"]
            .sum()
            .sort_values("sales", ascending=False)
            .head(15)
        )
        fig = px.bar(
            family_totals,
            x="sales",
            y="family",
            orientation="h",
            title="Top 15 families by total sales",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width='stretch')


def render_forecast(filtered: pd.DataFrame, run_metrics: pd.DataFrame) -> None:
    st.subheader("Forecast vs actual (validation window)")
    if filtered.empty:
        st.warning("No rows match the current filters.")
        return

    daily = daily_totals(filtered)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["actual"],
            mode="lines+markers",
            name="Actual",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["forecast"],
            mode="lines+markers",
            name="LightGBM forecast",
        )
    )
    if "baseline" in daily.columns:
        fig.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["baseline"],
                mode="lines",
                name="Seasonal naive",
                line={"dash": "dot"},
            )
        )
    fig.update_layout(xaxis_title="Date", yaxis_title="Total sales", hovermode="x unified")
    st.plotly_chart(fig, width='stretch')


def render_breakdowns(filtered: pd.DataFrame) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Error by family")
        summary = family_summary(filtered).head(15)
        fig = px.bar(
            summary,
            x="mean_abs_error",
            y="family",
            orientation="h",
            title="Mean absolute error by family",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width='stretch')
    with right:
        st.subheader("Error by store")
        summary = store_summary(filtered).head(15)
        summary["store_nbr"] = summary["store_nbr"].astype(str)
        fig = px.bar(
            summary,
            x="mean_abs_error",
            y="store_nbr",
            orientation="h",
            title="Mean absolute error by store",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width='stretch')


def render_metrics_table(metrics: pd.DataFrame) -> None:
    st.subheader("Model comparison")
    display = metrics.drop(columns=["cutoff"], errors="ignore").copy()
    for column in ("mae", "rmse", "rmsle"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: f"{value:,.4f}")
    st.dataframe(display, width='stretch')
    st.caption(
        "The LightGBM model and the seasonal-naive baseline are scored on the "
        "same 16-day validation window with identical metrics."
    )


def render_empty_state() -> None:
    st.title("Retail Sales Forecasting")
    st.warning(
        "The Kaggle dataset was not found in data/raw/. Download it first by "
        "following the README instructions, then reload this page."
    )
    st.markdown(
        "Required files: `train.csv`, `test.csv`, `stores.csv`, `oil.csv`, "
        "`holidays_events.csv`, and `transactions.csv` in `data/raw/`."
    )


def main() -> None:
    if "train" not in available_datasets():
        render_empty_state()
        return

    datasets = load_historical()
    predictions, metrics = load_forecast_cached()

    families, stores, date_range = render_filters(predictions)
    filtered = filter_predictions(
        predictions,
        families=families,
        stores=stores,
        date_range=date_range,
    )

    render_header(datasets)
    render_kpis(filtered, metrics)

    tab_trends, tab_forecast, tab_errors, tab_metrics = st.tabs(
        ["Sales trends", "Forecast", "Error analysis", "Model comparison"]
    )
    with tab_trends:
        render_trends(datasets)
    with tab_forecast:
        render_forecast(filtered, metrics)
    with tab_errors:
        render_breakdowns(filtered)
    with tab_metrics:
        render_metrics_table(metrics)


if __name__ == "__main__":
    main()

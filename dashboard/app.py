
import pandas as pd

import sys
from pathlib import Path

# Add SalesPulse project root to Python import path
BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests

from src.prediction.custom_dataset import (
    discover_models,
    validate_dataset,
    save_dataset,
    list_datasets,
    get_dataset_predictions,
    delete_dataset,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SalesPulse | Retail Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000"


# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #e5e7eb;
}

header[data-testid="stHeader"] {
    background: transparent;
}

footer, #MainMenu {
    visibility: hidden;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

section[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263247;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
}
section[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 0.35rem;
}
section[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 0.35rem 0.5rem;
    border-radius: 8px;
}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #1e293b;
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

h1, h2, h3 {
    color: #f8fafc !important;
    letter-spacing: -0.5px;
}

p, label {
    color: #cbd5e1;
}

.hero {
    background: linear-gradient(120deg, #172554, #1e3a5f, #164e63);
    border: 1px solid #334e75;
    padding: 30px 34px;
    border-radius: 18px;
    margin-bottom: 25px;
}

.hero-title {
    color: #ffffff;
    font-size: 34px;
    font-weight: 750;
}

.hero-subtitle {
    color: #cbd5e1;
    font-size: 15px;
    margin-top: 9px;
}

.kpi-card {
    background: #151f32;
    border: 1px solid #29364c;
    border-radius: 15px;
    padding: 20px;
    min-height: 125px;
}

.kpi-label {
    color: #94a3b8;
    font-size: 13px;
}

.kpi-value {
    color: #f8fafc;
    font-size: 28px;
    font-weight: 750;
    margin-top: 12px;
}

.kpi-caption {
    color: #64748b;
    font-size: 12px;
    margin-top: 5px;
}

.section-heading {
    color: #f8fafc;
    font-size: 22px;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 5px;
}

.section-description {
    color: #94a3b8;
    font-size: 13px;
    margin-bottom: 18px;
}

.status-pill {
    display: inline-block;
    color: #6ee7b7;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.25);
    padding: 7px 13px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 600;
}

[data-testid="stDataFrame"] {
    border: 1px solid #29364c;
    border-radius: 12px;
}

div[data-baseweb="select"] > div {
    background: #151f32;
    border-color: #334155;
    border-radius: 9px;
}

.stDownloadButton button {
    background: #2563eb;
    color: white;
    border: 0;
    border-radius: 9px;
    font-weight: 600;
}

.stDownloadButton button:hover {
    background: #1d4ed8;
    color: white;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding-top: 30px;
    border-top: 1px solid #263247;
    margin-top: 35px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CHART HELPERS
# ============================================================

def style_chart(fig, height=350):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Arial",
            color="#cbd5e1",
            size=12,
        ),
        margin=dict(l=15, r=15, t=35, b=15),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            y=1.12,
        ),
        xaxis=dict(gridcolor="#263247"),
        yaxis=dict(gridcolor="#263247"),
    )
    return fig


def show_kpi(column, label, value, caption):
    with column:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-caption">{caption}</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# API FUNCTIONS FOR THE ORIGINAL M5 DASHBOARD
# ============================================================

@st.cache_data(ttl=30)
def load_m5_predictions():
    response = requests.get(
        f"{API_URL}/predict",
        params={"limit": 50000},
        timeout=60,
    )
    response.raise_for_status()
    return pd.DataFrame(response.json()["predictions"])


@st.cache_data(ttl=30)
def load_model_metrics():
    response = requests.get(
        f"{API_URL}/metrics",
        timeout=30,
    )
    response.raise_for_status()
    return pd.DataFrame(response.json()["model_metrics"])


@st.cache_data(ttl=60)
def load_28day_evaluation():
    results_dir = (
        BASE_DIR / "data" / "processed" / "model_results"
    )

    summary_file = results_dir / "recursive_28day_summary.csv"
    daily_file = results_dir / "recursive_28day_daily_metrics.csv"
    predictions_file = results_dir / "recursive_28day_predictions.csv"

    required_files = [
        summary_file,
        daily_file,
        predictions_file,
    ]

    missing = [str(path) for path in required_files if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing 28-day evaluation files: " + ", ".join(missing)
        )

    summary = pd.read_csv(summary_file)
    daily = pd.read_csv(daily_file)
    predictions = pd.read_csv(predictions_file)

    daily["date"] = pd.to_datetime(daily["date"])
    predictions["date"] = pd.to_datetime(predictions["date"])

    return summary, daily, predictions


@st.cache_data(ttl=15)
def check_api():
    response = requests.get(
        f"{API_URL}/health",
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div style="padding: 12px 0 25px 0;">
        <div style="font-size: 27px; font-weight: 800; color: #f8fafc;">
            📊 SalesPulse
        </div>
        <div style="font-size: 12px; color: #94a3b8;">
            RETAIL INTELLIGENCE PLATFORM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("CSCE 5210 | Fundamentals of AI")
    st.caption("Retail sales forecasting prototype")


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">SalesPulse</div>
    <div class="hero-subtitle">
        AI-powered retail sales prediction and demand intelligence.
        Explore M5 results or upload your own historical sales data.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# MAIN DASHBOARD TABS
# ============================================================

page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "M5 Forecasts", "Forecast Evaluation", "Upload & Predict",
     "My Datasets", "Model Comparison", "About & Settings"],
    key="main_navigation",
)


# ============================================================
# TAB 1 — ORIGINAL M5 DASHBOARD
# ============================================================

if page == "M5 Forecasts":

    st.subheader("M5 Model Predictions")

    try:
        api_status = check_api()

        if api_status.get("status") != "healthy":
            st.error("The FastAPI backend is not healthy.")
            st.stop()

        predictions = load_m5_predictions()
        metrics = load_model_metrics()

    except requests.RequestException as error:
        st.error(
            "Unable to connect to FastAPI. Start it in another terminal:\n\n"
            "uvicorn api.main:app --reload"
        )
        st.code(str(error))
        predictions = pd.DataFrame()
        metrics = pd.DataFrame()

    if not predictions.empty:

        required_columns = [
            "id",
            "date",
            "sales",
            "predicted_sales",
            "model_name",
        ]

        missing = [
            col for col in required_columns
            if col not in predictions.columns
        ]

        if missing:
            st.error(f"API prediction response is missing: {missing}")
        else:
            predictions["date"] = pd.to_datetime(
                predictions["date"],
                errors="coerce",
            )
            predictions["sales"] = pd.to_numeric(
                predictions["sales"],
                errors="coerce",
            )
            predictions["predicted_sales"] = pd.to_numeric(
                predictions["predicted_sales"],
                errors="coerce",
            )

            predictions = predictions.dropna(
                subset=["date", "sales", "predicted_sales"]
            )

            product_ids = sorted(
                predictions["id"].dropna().unique()
            )

            selected_product = st.selectbox(
                "Filter by product/store",
                ["All Products"] + product_ids,
                key="m5_product_filter",
            )

            if selected_product == "All Products":
                filtered = predictions.copy()
            else:
                filtered = predictions[
                    predictions["id"] == selected_product
                ].copy()

            k1, k2, k3, k4 = st.columns(4)

            show_kpi(
                k1,
                "PRODUCT-STORE SERIES",
                f"{filtered['id'].nunique():,}",
                "Unique product-store combinations",
            )
            show_kpi(
                k2,
                "ACTUAL SALES",
                f"{filtered['sales'].sum():,.0f}",
                "Observed sales units",
            )
            show_kpi(
                k3,
                "PREDICTED SALES",
                f"{filtered['predicted_sales'].sum():,.0f}",
                "Estimated sales units",
            )
            show_kpi(
                k4,
                "AVERAGE PREDICTION",
                f"{filtered['predicted_sales'].mean():.2f}",
                "Estimated units per record",
            )

            st.markdown(
                '<div class="section-heading">Actual vs Predicted</div>',
                unsafe_allow_html=True,
            )

            chart_data = (
                filtered.groupby("id", as_index=False)
                .agg(
                    actual_sales=("sales", "sum"),
                    predicted_sales=("predicted_sales", "sum"),
                )
                .head(15)
            )

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=chart_data["id"],
                y=chart_data["actual_sales"],
                name="Actual Sales",
            ))

            fig.add_trace(go.Bar(
                x=chart_data["id"],
                y=chart_data["predicted_sales"],
                name="Predicted Sales",
            ))

            fig.update_layout(
                barmode="group",
                xaxis_title="Product / Store",
                yaxis_title="Sales Units",
                xaxis_tickangle=-35,
            )

            st.plotly_chart(
                style_chart(fig, 400),
                use_container_width=True,
            )

            st.markdown(
                '<div class="section-heading">Prediction Explorer</div>',
                unsafe_allow_html=True,
            )

            search_text = st.text_input(
                "Search product/store ID",
                key="m5_search",
            )

            table_data = filtered.copy()

            if search_text:
                table_data = table_data[
                    table_data["id"].astype(str).str.contains(
                        search_text,
                        case=False,
                        na=False,
                    )
                ]

            st.dataframe(
                table_data[required_columns].head(500),
                use_container_width=True,
                hide_index=True,
                height=350,
            )

            st.download_button(
                "Download M5 Predictions",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="salespulse_m5_predictions.csv",
                mime="text/csv",
                key="download_m5_predictions",
            )

    else:
        st.info("M5 dashboard data is not currently available.")

    st.markdown(
        '<div class="section-heading">Model Performance Comparison</div>',
        unsafe_allow_html=True,
    )

    if not metrics.empty:

        st.dataframe(
            metrics,
            use_container_width=True,
            hide_index=True,
        )

        model_column = next(
            (
                column for column in metrics.columns
                if column.lower() in ["model", "model_name", "name"]
            ),
            None,
        )

        metric_options = [
            column for column in metrics.select_dtypes(
                include="number"
            ).columns
            if column.lower() in [
                "mae",
                "rmse",
                "mape",
                "wape",
                "mape_nonzero_percent",
                "wape_percent",
            ]
        ]

        if model_column and metric_options:

            selected_metric = st.selectbox(
                "Select evaluation metric",
                metric_options,
                key="m5_metric",
            )

            comparison = metrics[
                [model_column, selected_metric]
            ].dropna().sort_values(
                selected_metric,
                ascending=True,
            )

            fig = px.bar(
                comparison,
                x=selected_metric,
                y=model_column,
                orientation="h",
                text=selected_metric,
            )

            fig.update_traces(
                texttemplate="%{text:.3f}",
                textposition="outside",
            )

            st.plotly_chart(
                style_chart(fig, 350),
                use_container_width=True,
            )
    else:
        st.info("No model metrics are currently available.")


# ============================================================
# TAB 2 — UPLOAD AND PREDICT
# ============================================================




if page == "Forecast Evaluation":
    # ========================================================
    # 28-DAY RECURSIVE FORECAST EVALUATION
    # ========================================================

    st.markdown("---")
    st.subheader("28-Day Recursive Forecast Evaluation")

    st.caption(
        "Out-of-sample evaluation from April 25 to May 22, 2016. "
        "Forecasts are generated recursively over 28 days."
    )

    try:
        summary_28, daily_28, predictions_28 = (
            load_28day_evaluation()
        )

        best_model = summary_28.sort_values(
            "RMSE"
        ).iloc[0]

        k1, k2, k3, k4 = st.columns(4)

        k1.metric(
            "Best Model (RMSE)",
            best_model["model"],
        )

        k2.metric(
            "Best RMSE",
            f"{best_model['RMSE']:.3f}",
        )

        k3.metric(
            "Forecast Horizon",
            f"{int(best_model['forecast_days'])} days",
        )

        k4.metric(
            "Product-Store Series",
            f"{int(best_model['product_series']):,}",
        )

        st.markdown("#### Overall Model Comparison")

        st.dataframe(
            summary_28.sort_values("RMSE"),
            use_container_width=True,
            hide_index=True,
        )

        metric_options = {
            "MAE": "MAE",
            "RMSE": "RMSE",
            "MAPE (%)": "MAPE_nonzero_percent",
            "WAPE (%)": "WAPE_percent",
        }

        selected_28_metric = st.selectbox(
            "Compare models using",
            options=list(metric_options.keys()),
            key="recursive_28day_metric",
        )

        metric_column = metric_options[selected_28_metric]

        comparison_28 = summary_28.sort_values(
            metric_column
        )

        fig = px.bar(
            comparison_28,
            x="model",
            y=metric_column,
            color="model",
            text=metric_column,
            title=f"28-Day Model Comparison — {selected_28_metric}",
        )

        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
        )

        st.plotly_chart(
            style_chart(fig, 380),
            use_container_width=True,
        )

        st.markdown("#### Forecast Accuracy by Day")

        selected_28_model = st.selectbox(
            "Select model",
            options=sorted(daily_28["model"].unique()),
            key="recursive_28day_model",
        )

        selected_daily = daily_28[
            daily_28["model"] == selected_28_model
        ].sort_values("forecast_day")

        fig = px.line(
            selected_daily,
            x="forecast_day",
            y="MAE",
            markers=True,
            title=f"Daily MAE — {selected_28_model}",
            labels={
                "forecast_day": "Forecast Day",
                "MAE": "Mean Absolute Error",
            },
        )

        st.plotly_chart(
            style_chart(fig, 350),
            use_container_width=True,
        )

        st.markdown("#### Actual vs Predicted Sales")

        available_products = sorted(
            predictions_28["id"].unique()
        )

        selected_28_product = st.selectbox(
            "Select product-store series",
            options=available_products,
            key="recursive_28day_product",
        )

        product_model_data = predictions_28[
            (predictions_28["id"] == selected_28_product)
            & (predictions_28["model"] == selected_28_model)
        ].sort_values("forecast_day")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=product_model_data["date"],
            y=product_model_data["actual_sales"],
            mode="lines+markers",
            name="Actual Sales",
        ))

        fig.add_trace(go.Scatter(
            x=product_model_data["date"],
            y=product_model_data["predicted_sales"],
            mode="lines+markers",
            name="Predicted Sales",
        ))

        fig.update_layout(
            title=f"28-Day Forecast — {selected_28_product}",
            xaxis_title="Forecast Date",
            yaxis_title="Sales Units",
        )

        st.plotly_chart(
            style_chart(fig, 380),
            use_container_width=True,
        )

        st.download_button(
            "Download 28-Day Evaluation Results",
            data=predictions_28.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="salespulse_28day_predictions.csv",
            mime="text/csv",
            key="download_28day_results",
        )

    except Exception as error:
        st.warning(
            f"28-day evaluation data could not be loaded: {error}"
        )

if page == "Upload & Predict":

    st.subheader("Upload Your Retail Sales Dataset")

    st.write(
        "Upload historical daily sales data and select one of "
        "your existing trained models to generate next-day predictions."
    )

    st.info(
        "Required CSV columns: **id**, **date**, and **sales**. "
        "Each product must have at least 29 consecutive daily records."
    )

    try:
        available_models = discover_models()

    except Exception as error:
        available_models = {}
        st.error(str(error))

    if available_models:

        selected_model = st.selectbox(
            "Select trained prediction model",
            options=list(available_models.keys()),
            index=0,
            help=(
                "The selected model will be used to generate predictions "
                "for every product in the uploaded CSV."
            ),
            key="custom_selected_model",
        )

        st.caption(
            f"Available trained models: {', '.join(available_models.keys())}"
        )

        uploaded_file = st.file_uploader(
            "Choose historical sales CSV",
            type=["csv"],
            key="custom_sales_csv",
        )

        if uploaded_file is not None:

            try:
                raw_df = pd.read_csv(uploaded_file)

                st.markdown("#### Uploaded Data Preview")

                st.dataframe(
                    raw_df.head(10),
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    f"{len(raw_df):,} uploaded rows | "
                    f"{len(raw_df.columns)} columns"
                )

                if st.button(
                    "Generate Predictions",
                    type="primary",
                    use_container_width=True,
                    key="run_custom_prediction",
                ):

                    with st.spinner(
                        f"Validating data and running {selected_model}..."
                    ):

                        clean_df = validate_dataset(raw_df)

                        metadata, result_df = save_dataset(
                            filename=uploaded_file.name,
                            dataframe=clean_df,
                            model_name=selected_model,
                        )

                    st.session_state["last_custom_dataset_id"] = (
                        metadata["dataset_id"]
                    )

                    st.success(
                        f"Successfully generated {len(result_df):,} "
                        "next-day predictions."
                    )

                    metric_cols = st.columns(3)

                    metric_cols[0].metric(
                        "Product series",
                        f"{metadata['products']:,}",
                    )

                    metric_cols[1].metric(
                        "Historical rows",
                        f"{metadata['rows']:,}",
                    )

                    metric_cols[2].metric(
                        "Forecast rows",
                        f"{metadata['forecast_rows']:,}",
                    )

                    st.markdown("#### Prediction Results")

                    st.dataframe(
                        result_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "Download Prediction CSV",
                        data=result_df.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="salespulse_custom_predictions.csv",
                        mime="text/csv",
                        key="download_new_custom_predictions",
                    )

                    st.caption(
                        f"Dataset ID: {metadata['dataset_id']} | "
                        f"Model: {metadata['model_name']}"
                    )

            except Exception as error:
                st.error(f"Could not generate predictions: {error}")


# ============================================================
# TAB 3 — MANAGE UPLOADED DATASETS
# ============================================================

if page == "My Datasets":

    st.subheader("Uploaded Dataset Library")

    st.write(
        "Review saved datasets, view their predictions, download results, "
        "or delete a dataset and its associated files."
    )

    datasets = list_datasets()

    if not datasets:

        st.info("No uploaded datasets have been saved yet.")

    else:

        dataset_table = pd.DataFrame([
            {
                "Dataset ID": item["dataset_id"],
                "File": item["filename"],
                "Model": item["model_name"],
                "Rows": item["rows"],
                "Products": item["products"],
                "Uploaded": item["uploaded_at"][:19].replace("T", " "),
            }
            for item in datasets
        ])

        st.dataframe(
            dataset_table,
            use_container_width=True,
            hide_index=True,
        )

        dataset_options = {
            (
                f"{item['filename']} | "
                f"{item['model_name']} | "
                f"{item['dataset_id']}"
            ): item["dataset_id"]
            for item in datasets
        }

        selected_dataset_label = st.selectbox(
            "Select saved dataset",
            options=list(dataset_options.keys()),
            key="manage_selected_dataset",
        )

        selected_dataset_id = dataset_options[
            selected_dataset_label
        ]

        view_col, delete_col = st.columns(2)

        with view_col:

            if st.button(
                "View Predictions",
                use_container_width=True,
                key="view_saved_predictions",
            ):

                try:
                    saved_predictions = get_dataset_predictions(
                        selected_dataset_id
                    )

                    st.session_state["saved_predictions_view"] = (
                        saved_predictions
                    )

                    st.session_state["saved_predictions_id"] = (
                        selected_dataset_id
                    )

                except Exception as error:
                    st.error(str(error))

        with delete_col:

            confirm_delete = st.checkbox(
                "I confirm that I want to delete this dataset",
                key=f"confirm_delete_{selected_dataset_id}",
            )

            if st.button(
                "Delete Selected Dataset",
                type="primary",
                disabled=not confirm_delete,
                use_container_width=True,
                key="delete_saved_dataset",
            ):

                try:
                    delete_dataset(selected_dataset_id)

                    st.session_state.pop(
                        "saved_predictions_view",
                        None,
                    )

                    st.session_state.pop(
                        "saved_predictions_id",
                        None,
                    )

                    st.success(
                        "Dataset, source CSV, and prediction CSV deleted."
                    )

                    st.rerun()

                except Exception as error:
                    st.error(f"Could not delete dataset: {error}")

        if (
            st.session_state.get("saved_predictions_id")
            == selected_dataset_id
            and "saved_predictions_view" in st.session_state
        ):

            st.markdown("#### Saved Prediction Results")

            saved_df = st.session_state[
                "saved_predictions_view"
            ]

            st.dataframe(
                saved_df,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download Saved Predictions",
                data=saved_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="salespulse_saved_predictions.csv",
                mime="text/csv",
                key="download_saved_predictions",
            )


if page == "Overview":
    st.markdown("## SalesPulse Overview")
    st.markdown(
        "A retail demand intelligence workspace for exploring historical sales, "
        "comparing machine-learning models, and generating product-level forecasts."
    )

    api_ok = False
    forecast_count = "—"
    model_count = "—"
    try:
        health = check_api()
        api_ok = health.get("status") == "healthy"
        overview_predictions = load_m5_predictions()
        forecast_count = f"{len(overview_predictions):,}"
        overview_metrics = load_model_metrics()
        model_col = next(
            (c for c in overview_metrics.columns
             if c.lower() in ["model", "model_name", "name"]),
            None,
        )
        model_count = (
            f"{overview_metrics[model_col].nunique():,}"
            if model_col else f"{len(overview_metrics):,}"
        )
    except Exception:
        pass

    st.markdown('<div class="section-heading">Platform at a glance</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    show_kpi(c1, "API STATUS", "Online" if api_ok else "Offline",
             "FastAPI prediction service")
    show_kpi(c2, "M5 FORECASTS", forecast_count,
             "Available prediction records")
    show_kpi(c3, "TRAINED MODELS", model_count,
             "Models available for comparison")
    show_kpi(c4, "FORECAST HORIZON", "1–28 days",
             "Next-day and recursive evaluation")

    st.markdown('<div class="section-heading">Explore SalesPulse</div>',
                unsafe_allow_html=True)
    cards = st.columns(3)
    destinations = [
        ("M5 Forecasts", "Explore product-store predictions and forecast totals."),
        ("Forecast Evaluation", "Compare recursive 28-day model performance."),
        ("Upload & Predict", "Upload a CSV and generate next-day predictions."),
        ("My Datasets", "Review, download, or delete saved datasets."),
        ("Model Comparison", "Compare model error metrics side by side."),
        ("About & Settings", "Review project details and data requirements."),
    ]
    for index, (heading, description) in enumerate(destinations):
        with cards[index % 3]:
            st.markdown(
                f"""<div class="kpi-card" style="min-height:112px;margin-bottom:14px;">
                    <div style="font-size:16px;font-weight:700;color:#f8fafc;">{heading}</div>
                    <div style="font-size:13px;color:#94a3b8;margin-top:9px;">{description}</div>
                </div>""",
                unsafe_allow_html=True,
            )


if page == "Model Comparison":
    st.subheader("Model Comparison")
    try:
        comparison_metrics = load_model_metrics()
        st.dataframe(comparison_metrics, use_container_width=True, hide_index=True)
        numeric = [c for c in comparison_metrics.select_dtypes(include="number").columns if c.lower() in ["mae", "rmse", "mape", "wape", "mape_nonzero_percent", "wape_percent"]]
        model_col = next((c for c in comparison_metrics.columns if c.lower() in ["model", "model_name", "name"]), None)
        if numeric and model_col:
            metric_choice = st.selectbox("Evaluation metric", numeric, key="sidebar_model_metric")
            fig = px.bar(comparison_metrics.sort_values(metric_choice), x=metric_choice, y=model_col, orientation="h", title=f"Model comparison — {metric_choice}")
            st.plotly_chart(style_chart(fig, 350), use_container_width=True)
    except Exception as error:
        st.warning(f"Model comparison is unavailable: {error}")

if page == "About & Settings":
    st.subheader("About & Settings")
    st.write("SalesPulse is a CSCE 5210 retail sales forecasting prototype.")
    st.markdown("**Stack:** Python, Pandas, NumPy, Scikit-learn, FastAPI, Streamlit, Plotly, and Joblib.\n\n**Custom CSV format:** `id`, `date`, `sales`.\n\n**Custom 28-day evaluation:** requires at least 57 consecutive observations per product and recursively feeds predictions into subsequent forecast steps.")


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    <strong>SalesPulse</strong> · AI-Powered Retail Sales Prediction System
    <br><br>
    CSCE 5210 · Fundamentals of Artificial Intelligence
    <br>
    Prototype version
</div>
""", unsafe_allow_html=True)


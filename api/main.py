
import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = (
    BASE_DIR / "data" / "processed" / "predictions"
    / "latest_predictions.csv"
)

METRICS_FILE = (
    BASE_DIR / "data" / "processed" / "model_results"
    / "model_comparison.csv"
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="SalesPulse API",
    description=(
        "AI-powered retail sales prediction API "
        "using machine learning models trained on M5 data."
    ),
    version="1.1.0",
)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return {
        "project": "SalesPulse",
        "description": "Retail Sales Prediction System",
        "status": "Running",
        "version": "1.1.0",
        "endpoints": {
            "predictions": "/predict",
            "metrics": "/metrics",
            "health": "/health",
            "documentation": "/docs",
        },
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "api": "running",
        "predictions_file_exists": PREDICTIONS_FILE.is_file(),
        "metrics_file_exists": METRICS_FILE.is_file(),
        "predictions_available": PREDICTIONS_FILE.is_file(),
        "metrics_available": METRICS_FILE.is_file(),
    }


# ============================================================
# PREDICTIONS
# ============================================================

@app.get("/predict")
def get_predictions(
    limit: int = Query(default=1000, ge=1, le=50000)
):

    if not PREDICTIONS_FILE.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                "Prediction file not found. "
                "Run the prediction module first."
            ),
        )

    try:
        data = pd.read_csv(PREDICTIONS_FILE, nrows=limit)

        if data.empty:
            return {
                "total_returned": 0,
                "predictions": [],
            }

        required_columns = {
            "id",
            "date",
            "sales",
            "predicted_sales",
            "model_name",
        }

        missing = required_columns.difference(data.columns)

        if missing:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Prediction file has an invalid schema. "
                    f"Missing columns: {sorted(missing)}"
                ),
            )

        data["date"] = pd.to_datetime(
            data["date"],
            errors="coerce",
        )

        if data["date"].isna().any():
            raise HTTPException(
                status_code=500,
                detail="Prediction file contains invalid dates.",
            )

        for column in ["sales", "predicted_sales"]:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

        if data[["sales", "predicted_sales"]].isna().any().any():
            raise HTTPException(
                status_code=500,
                detail="Prediction file contains invalid sales values.",
            )

        records = json.loads(
            data.to_json(
                orient="records",
                date_format="iso",
            )
        )

        return {
            "total_returned": len(records),
            "predictions": records,
        }

    except HTTPException:
        raise

    except (OSError, pd.errors.ParserError, ValueError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read prediction file: {error}",
        )


# ============================================================
# MODEL METRICS
# ============================================================

@app.get("/metrics")
def get_model_metrics():

    if not METRICS_FILE.is_file():
        raise HTTPException(
            status_code=404,
            detail="Model comparison file not found. Train models first.",
        )

    try:
        data = pd.read_csv(METRICS_FILE)

        if data.empty:
            return {
                "total_models": 0,
                "model_metrics": [],
            }

        records = json.loads(
            data.to_json(orient="records")
        )

        return {
            "total_models": len(records),
            "model_metrics": records,
        }

    except (OSError, pd.errors.ParserError, ValueError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load model metrics: {error}",
        )

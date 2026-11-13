
import gc
import json
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
FEATURE_DIR = BASE_DIR / "data" / "processed" / "features"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "predictions"


def load_best_model():
    """Load the selected model and its metadata."""

    metadata_path = MODEL_DIR / "best_model.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    best_model = (
        metadata.get("best_model")
        or metadata.get("model_name")
    )

    if not best_model:
        raise ValueError(
            "best_model.json does not contain a model name."
        )

    model_mapping = {
        "Random Forest": "random_forest",
        "Linear Regression": "linear_regression",
        "HistGradientBoosting": "histgradientboosting",
    }

    model_name = model_mapping.get(
        best_model,
        best_model.lower().replace(" ", "_")
    )

    model_path = MODEL_DIR / f"{model_name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    print(f"Loading model: {model_name}")

    model = joblib.load(model_path)

    return model, model_name


def load_latest_features_batch():
    """
    Read feature files one at a time and retain the latest
    record for every original product-store series.
    """

    files = sorted(FEATURE_DIR.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in {FEATURE_DIR}"
        )

    latest_batches = []

    print(f"Found {len(files)} feature files.")
    print("Collecting latest product-store records...")

    for index, file in enumerate(files, start=1):

        print(f"Processing {index}/{len(files)}: {file.name}")

        batch = pd.read_parquet(file)

        batch["date"] = pd.to_datetime(
            batch["date"],
            errors="coerce"
        )

        batch = batch.dropna(subset=["date"])

        batch = batch.sort_values("date")

        latest = (
            batch.groupby("id", sort=False)
            .tail(1)
            .copy()
        )

        latest_batches.append(latest)

        del batch
        del latest

        gc.collect()

    result = pd.concat(
        latest_batches,
        ignore_index=True
    )

    del latest_batches
    gc.collect()

    print(f"Latest records collected: {len(result):,}")

    return result


def predict_sales():
    """Generate sales predictions for all product-store series."""

    model, model_name = load_best_model()

    latest = load_latest_features_batch()

    excluded_columns = {
        "id",
        "date",
        "sales",
    }

    # Use the exact training feature order when available.
    if hasattr(model, "feature_names_in_"):
        feature_columns = list(model.feature_names_in_)
    else:
        feature_columns = [
            column
            for column in latest.columns
            if column not in excluded_columns
        ]

    missing_columns = [
        column
        for column in feature_columns
        if column not in latest.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing model features: {missing_columns}"
        )

    X = latest[feature_columns]

    print("Generating predictions...")

    predictions = model.predict(X)

    results = latest[
        ["id", "date", "sales"]
    ].copy()

    # Preserve the original M5 ID for traceability.
    results["source_id"] = results["id"]

    # Remove the dataset split suffix for cleaner display.
    results["id"] = (
        results["id"]
        .astype(str)
        .str.replace(r"_evaluation$", "", regex=True)
    )

    # Ensure predictions are non-negative.
    results["predicted_sales"] = (
        pd.Series(predictions)
        .clip(lower=0)
        .to_numpy()
    )

    results["model_name"] = model_name

    # Display dates consistently as YYYY-MM-DD.
    results["date"] = (
        pd.to_datetime(results["date"])
        .dt.strftime("%Y-%m-%d")
    )

    results = results.sort_values(
        by="id"
    ).reset_index(drop=True)

    return results


if __name__ == "__main__":

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Starting SalesPulse predictions...")

    predictions = predict_sales()

    output_file = OUTPUT_DIR / "latest_predictions.csv"

    predictions.to_csv(
        output_file,
        index=False
    )

    print("\nPrediction completed successfully.")
    print(f"Total predictions: {len(predictions):,}")
    print(f"Output file: {output_file}")

    print("\nSample predictions:")
    print(
        predictions[
            [
                "id",
                "date",
                "sales",
                "predicted_sales",
                "model_name"
            ]
        ].head(10).to_string(index=False)
    )

    del predictions
    gc.collect()

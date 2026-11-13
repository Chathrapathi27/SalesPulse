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

    best_model = metadata.get("best_model") or metadata.get("model_name")

    if not best_model:
        raise ValueError("best_model.json does not contain a model name.")

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
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model: {model_name}")

    return joblib.load(model_path), model_name


def load_latest_features_batch(feature_columns):
    """
    Read only the columns required for prediction.

    Each Parquet file contains different product-store series
    across the full historical date range.
    """

    files = sorted(FEATURE_DIR.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in {FEATURE_DIR}"
        )

    required_columns = list(dict.fromkeys(
        ["id", "date", "sales"] + list(feature_columns)
    ))

    latest_batches = []

    print(f"Found {len(files)} feature files.")
    print("Reading only required prediction columns...")

    for index, file in enumerate(files, start=1):
        print(f"Processing {index}/{len(files)}: {file.name}")

        try:
            batch = pd.read_parquet(
                file,
                columns=required_columns
            )
        except Exception as error:
            raise ValueError(
                f"Could not read required columns from {file.name}: {error}"
            ) from error

        batch["date"] = pd.to_datetime(
            batch["date"],
            errors="coerce"
        )

        batch = batch.dropna(subset=["id", "date"])

        if batch.empty:
            del batch
            continue

        # Select the most recent observation for each product series.
        latest_indices = (
            batch.groupby("id", sort=False)["date"].idxmax()
        )

        latest = batch.loc[latest_indices].copy()

        latest_batches.append(latest)

        del batch, latest, latest_indices

        if index % 10 == 0:
            gc.collect()

    if not latest_batches:
        raise ValueError("No valid product records were found.")

    result = pd.concat(
        latest_batches,
        ignore_index=True
    )

    del latest_batches
    gc.collect()

    # Each product-store series should belong to exactly one file.
    duplicate_ids = result["id"].duplicated()

    if duplicate_ids.any():
        examples = result.loc[duplicate_ids, "id"].head(5).tolist()

        raise ValueError(
            "Duplicate product series found across feature files. "
            f"Examples: {examples}"
        )

    print(f"Latest product records collected: {len(result):,}")

    return result


def predict_sales():
    """Generate one-step-ahead predictions for all product-store series."""

    model, model_name = load_best_model()

    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "The selected model does not expose feature_names_in_. "
            "Please check the saved model."
        )

    feature_columns = list(model.feature_names_in_)

    latest = load_latest_features_batch(feature_columns)

    missing_columns = [
        column for column in feature_columns
        if column not in latest.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing model features: {missing_columns}")

    X = latest[feature_columns]

    print("Generating predictions...")

    predictions = model.predict(X)

    results = latest[["id", "date", "sales"]].copy()

    # Preserve original M5 IDs for traceability.
    results["source_id"] = results["id"]

    # Remove the evaluation suffix for dashboard display.
    results["id"] = (
        results["id"]
        .astype(str)
        .str.replace(r"_evaluation$", "", regex=True)
    )

    results["predicted_sales"] = (
        pd.Series(predictions, index=results.index)
        .clip(lower=0)
        .to_numpy()
    )

    results["model_name"] = model_name

    results["date"] = (
        pd.to_datetime(results["date"])
        .dt.strftime("%Y-%m-%d")
    )

    results = results.sort_values(
        by="id"
    ).reset_index(drop=True)

    return results


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Starting SalesPulse predictions...")

    predictions = predict_sales()

    output_file = OUTPUT_DIR / "latest_predictions.csv"

    predictions.to_csv(output_file, index=False)

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

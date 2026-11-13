
import json
import re
import shutil
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "data" / "user_datasets"

REQUIRED_COLUMNS = ["id", "date", "sales"]
MIN_HISTORY_DAYS = 29

# Known filenames for the models trained in SalesPulse.
MODEL_LABELS = {
    "random_forest": "Random Forest",
    "linear_regression": "Linear Regression",
    "histgradientboosting": "HistGradientBoosting",
}


def discover_models():
    """Find available trained model files in the models directory."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    available = {}

    for model_key, label in MODEL_LABELS.items():
        model_path = MODEL_DIR / f"{model_key}.joblib"

        if model_path.exists():
            available[label] = model_key

    if not available:
        raise FileNotFoundError(
            f"No trained .joblib models were found in {MODEL_DIR}. "
            "Train the models before using custom dataset predictions."
        )

    return available


def load_model(model_name):
    """Load a selected trained model."""

    available = discover_models()

    if model_name not in available:
        raise ValueError(
            f"Model '{model_name}' is not available. "
            f"Available models: {', '.join(available)}"
        )

    model_key = available[model_name]
    model_path = MODEL_DIR / f"{model_key}.joblib"

    model = joblib.load(model_path)

    if not hasattr(model, "predict"):
        raise ValueError(
            f"The saved file for {model_name} is not a prediction model."
        )

    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            f"{model_name} does not contain feature_names_in_. "
            "Please retrain and save the model with named DataFrame features."
        )

    return model, model_key



def validate_dataset(dataframe):
    """
    Validate and normalize an uploaded historical retail sales CSV.

    Required columns:
        id, date, sales

    Validation:
        - Non-empty dataset
        - Required columns
        - Valid product IDs
        - Valid dates
        - Numeric, finite, non-negative sales
        - No duplicate product/date combinations
        - At least 29 consecutive daily observations per product
    """

    if dataframe is None:
        raise ValueError("No dataset was provided.")

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Uploaded data must be a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("The uploaded CSV is empty.")

    df = dataframe.copy()

    # Normalize column names.
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
    )

    # Detect duplicate column names after normalization.
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(
            f"Duplicate column names detected: {duplicates}"
        )

    # Check required columns.
    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}. "
            "Required columns are: id, date, sales."
        )

    # Keep only the required fields for this prototype.
    df = df[REQUIRED_COLUMNS].copy()

    # Validate IDs.
    df["id"] = df["id"].astype("string").str.strip()

    if df["id"].isna().any() or df["id"].eq("").any():
        raise ValueError(
            "Product IDs cannot be empty or missing."
        )


    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        format="mixed",
    )

    if df["date"].isna().any():
        bad_rows = (
            df.index[df["date"].isna()].tolist()[:10]
        )

        raise ValueError(
            "Invalid or missing dates detected. "
            f"Example row indexes: {bad_rows}"
        )

    # Normalize to daily dates.
    if (df["date"] != df["date"].dt.normalize()).any():
        raise ValueError(
            "Dates must represent whole days without time values."
        )

    # Validate sales.
    df["sales"] = pd.to_numeric(
        df["sales"],
        errors="coerce",
    )

    if df["sales"].isna().any():
        raise ValueError(
            "Sales must contain valid numeric values. "
            "Empty cells and non-numeric values are not allowed."
        )

    sales_values = df["sales"].to_numpy(dtype=float)

    if not np.isfinite(sales_values).all():
        raise ValueError(
            "Sales values must be finite. "
            "NaN and infinite values are not allowed."
        )

    if (df["sales"] < 0).any():
        raise ValueError(
            "Negative sales values are not allowed."
        )

    # Reject duplicate product/date records instead of silently
    # adding them together.
    duplicate_mask = df.duplicated(
        subset=["id", "date"],
        keep=False,
    )

    if duplicate_mask.any():
        examples = (
            df.loc[duplicate_mask, ["id", "date"]]
            .head(5)
            .to_dict(orient="records")
        )

        raise ValueError(
            "Duplicate product/date records detected. "
            f"Examples: {examples}. "
            "Please keep only one sales record per product per day."
        )

    # Sort records.
    df = (
        df.sort_values(["id", "date"])
        .reset_index(drop=True)
    )

    # Validate each product's daily history.
    for product_id, history in df.groupby("id", sort=False):

        dates = pd.DatetimeIndex(history["date"])

        if len(dates) < MIN_HISTORY_DAYS:
            raise ValueError(
                f"Product '{product_id}' has only {len(dates)} "
                f"observations. At least {MIN_HISTORY_DAYS} "
                "daily observations are required."
            )

        # Check all dates, not just the final 29.
        expected_dates = pd.date_range(
            start=dates.min(),
            end=dates.max(),
            freq="D",
        )

        missing_dates = expected_dates.difference(dates)

        if len(missing_dates) > 0:
            examples = [
                date.strftime("%Y-%m-%d")
                for date in missing_dates[:5]
            ]

            raise ValueError(
                f"Product '{product_id}' has missing daily dates. "
                f"Examples: {examples}. "
                "Add the missing dates with zero sales if appropriate."
            )

    # Ensure consistent data types.
    df["id"] = df["id"].astype(str)
    df["sales"] = df["sales"].astype(float)

    return df



def _make_feature_row(history, feature_columns):
    """Create one feature row using the selected model's expected columns."""

    history = history.sort_values("date").reset_index(drop=True)

    sales = history["sales"].astype(float)
    last_date = pd.Timestamp(history["date"].iloc[-1])
    forecast_date = last_date + pd.Timedelta(days=1)

    weekday_number = forecast_date.day_of_week

    known_features = {
        # Lag features
        "lag_1": sales.iloc[-1],
        "lag_7": sales.iloc[-7],
        "lag_14": sales.iloc[-14],
        "lag_28": sales.iloc[-28],

        # Rolling features; only historical observations are used.
        "rolling_mean_7": sales.tail(7).mean(),
        "rolling_mean_28": sales.tail(28).mean(),

        # Calendar features
        "day_of_week": weekday_number,
        "is_weekend": int(weekday_number >= 5),
        "weekday": forecast_date.strftime("%A"),
        "wday": weekday_number + 1,
        "day": forecast_date.day,
        "month": forecast_date.month,
        "year": forecast_date.year,
        "wm_yr_wk": 0,

        # M5 event/calendar fields unavailable in a basic CSV.
        "event_name_1": "Unknown",
        "event_type_1": "Unknown",
        "event_name_2": "Unknown",
        "event_type_2": "Unknown",

        # SNAP fields are not supplied by the basic upload format.
        "snap_CA": 0,
        "snap_TX": 0,
        "snap_WI": 0,
    }

    row = {}

    for column in feature_columns:

        if column in known_features:
            row[column] = known_features[column]

        elif column.startswith("snap_"):
            row[column] = 0

        else:
            # Neutral default for other calendar or numeric fields.
            # The model pipeline must support these input columns.
            row[column] = 0

    return pd.DataFrame([row], columns=feature_columns), forecast_date


def generate_predictions(dataframe, model_name):
    """Validate input and generate one next-day prediction per product."""

    dataframe = validate_dataset(dataframe)

    model, model_key = load_model(model_name)
    feature_columns = list(model.feature_names_in_)

    prediction_rows = []

    for product_id, history in dataframe.groupby("id", sort=False):

        X, forecast_date = _make_feature_row(
            history,
            feature_columns
        )

        prediction = float(model.predict(X)[0])

        prediction_rows.append({
            "id": str(product_id),
            "last_observed_date": pd.Timestamp(
                history["date"].max()
            ).strftime("%Y-%m-%d"),
            "forecast_date": forecast_date.strftime("%Y-%m-%d"),
            "last_observed_sales": float(
                history["sales"].iloc[-1]
            ),
            "predicted_sales": max(0.0, prediction),
            "model_name": model_key,
            "model_display_name": model_name,
        })

    return pd.DataFrame(prediction_rows)


def save_dataset(filename, dataframe, model_name):
    """Validate, predict, and save an uploaded dataset."""

    dataframe = validate_dataset(dataframe)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    dataset_id = uuid.uuid4().hex[:12]
    dataset_folder = DATASET_DIR / dataset_id
    dataset_folder.mkdir(parents=True, exist_ok=False)

    safe_filename = re.sub(
        r"[^a-zA-Z0-9_.-]",
        "_",
        Path(filename).name
    )

    try:
        predictions = generate_predictions(
            dataframe,
            model_name
        )

        dataframe.to_csv(
            dataset_folder / "source.csv",
            index=False
        )

        predictions.to_csv(
            dataset_folder / "predictions.csv",
            index=False
        )

        metadata = {
            "dataset_id": dataset_id,
            "filename": safe_filename,
            "model_name": model_name,
            "rows": int(len(dataframe)),
            "products": int(dataframe["id"].nunique()),
            "forecast_rows": int(len(predictions)),
            "uploaded_at": pd.Timestamp.now().isoformat(),
        }

        with open(
            dataset_folder / "metadata.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(metadata, file, indent=4)

    except Exception:
        shutil.rmtree(dataset_folder, ignore_errors=True)
        raise

    return metadata, predictions


def list_datasets():
    """Return metadata for saved uploaded datasets."""

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for folder in DATASET_DIR.iterdir():

        metadata_path = folder / "metadata.json"

        if not folder.is_dir() or not metadata_path.exists():
            continue

        try:
            with open(
                metadata_path,
                "r",
                encoding="utf-8"
            ) as file:
                results.append(json.load(file))

        except (OSError, json.JSONDecodeError):
            continue

    return sorted(
        results,
        key=lambda item: item.get("uploaded_at", ""),
        reverse=True
    )


def get_dataset_predictions(dataset_id):
    """Read saved predictions for one dataset."""

    if not re.fullmatch(r"[a-f0-9]{12}", str(dataset_id)):
        raise ValueError("Invalid dataset ID.")

    folder = DATASET_DIR / dataset_id
    prediction_file = folder / "predictions.csv"

    if not prediction_file.is_file():
        raise FileNotFoundError("Dataset predictions were not found.")

    return pd.read_csv(prediction_file)


def delete_dataset(dataset_id):
    """Delete one uploaded dataset and its associated predictions."""

    if not re.fullmatch(r"[a-f0-9]{12}", str(dataset_id)):
        raise ValueError("Invalid dataset ID.")

    folder = DATASET_DIR / dataset_id

    if not folder.is_dir():
        raise FileNotFoundError("Dataset was not found.")

    # Resolve path and confirm it is inside the user dataset directory.
    root = DATASET_DIR.resolve()
    target = folder.resolve()

    if root not in target.parents:
        raise ValueError("Invalid dataset location.")

    shutil.rmtree(target)

    return True

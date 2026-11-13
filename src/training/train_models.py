
import os
import glob
import gc
import json
import joblib
import pandas as pd
import numpy as np

from tqdm import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

# ==================================================
# SALESPULSE - MODEL TRAINING
# Memory-conscious configuration for 8 GB RAM
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

FEATURE_DIR = os.path.join(
    BASE_DIR, "data", "processed", "features"
)

MODEL_DIR = os.path.join(
    BASE_DIR, "models"
)

RESULT_DIR = os.path.join(
    BASE_DIR, "data", "processed", "model_results"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Approximately 100 product-store series
SAMPLE_MODULO = 30490
SAMPLE_LIMIT = 100

# Last 28 days are reserved for validation
VALIDATION_DAYS = 28

# ==================================================
# COLUMNS
# ==================================================

READ_COLUMNS = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "date",
    "sales",
    "wday",
    "month",
    "year",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "day_of_week",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28"
]

CATEGORICAL_FEATURES = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id"
]

NUMERIC_FEATURES = [
    "wday",
    "month",
    "year",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "day_of_week",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28"
]

FEATURE_COLUMNS = (
    CATEGORICAL_FEATURES + NUMERIC_FEATURES
)

TARGET = "sales"


# ==================================================
# METRICS
# ==================================================

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual, predicted
    )

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    # MAPE is calculated only for nonzero actual values.
    nonzero_mask = actual != 0

    if nonzero_mask.sum() > 0:
        mape = np.mean(
            np.abs(
                (actual[nonzero_mask] -
                 predicted[nonzero_mask]) /
                actual[nonzero_mask]
            )
        ) * 100
    else:
        mape = np.nan

    # WAPE is also useful for intermittent retail demand.
    total_actual = np.abs(actual).sum()

    if total_actual > 0:
        wape = (
            np.abs(actual - predicted).sum()
            / total_actual
        ) * 100
    else:
        wape = np.nan

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE_nonzero_percent": float(mape),
        "WAPE_percent": float(wape)
    }


# ==================================================
# LOAD A REPRODUCIBLE SAMPLE
# ==================================================

def load_training_sample():

    files = sorted(
        glob.glob(
            os.path.join(
                FEATURE_DIR,
                "features_*.parquet"
            )
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No feature files found: {FEATURE_DIR}"
        )

    print("=" * 60)
    print("SALESPULSE - LOADING TRAINING SAMPLE")
    print("=" * 60)

    print("Feature files:", len(files))

    sampled_parts = []

    for file_path in tqdm(
        files,
        desc="Reading feature files"
    ):

        df = pd.read_parquet(
            file_path,
            columns=READ_COLUMNS
        )

        # Deterministic selection of product-store series.
        hashes = pd.util.hash_pandas_object(
            df["id"],
            index=False
        )

        selected = df[
            (hashes % SAMPLE_MODULO) < SAMPLE_LIMIT
        ].copy()

        if not selected.empty:
            sampled_parts.append(selected)

        del df
        del selected
        gc.collect()

    if not sampled_parts:
        raise ValueError(
            "No records selected. Check the sample configuration."
        )

    data = pd.concat(
        sampled_parts,
        ignore_index=True
    )

    del sampled_parts
    gc.collect()

    data["date"] = pd.to_datetime(data["date"])

    print("\nSample loaded successfully.")
    print("Rows:", f"{len(data):,}")
    print(
        "Product-store series:",
        data["id"].nunique()
    )
    print(
        "Date range:",
        data["date"].min(),
        "to",
        data["date"].max()
    )

    return data


# ==================================================
# CHRONOLOGICAL TRAIN / VALIDATION SPLIT
# ==================================================

def split_data(data):

    max_date = data["date"].max()

    validation_start = (
        max_date -
        pd.Timedelta(days=VALIDATION_DAYS - 1)
    )

    train_data = data[
        data["date"] < validation_start
    ].copy()

    validation_data = data[
        data["date"] >= validation_start
    ].copy()

    print("\nTraining period:")
    print(
        train_data["date"].min(),
        "to",
        train_data["date"].max()
    )

    print("\nValidation period:")
    print(
        validation_data["date"].min(),
        "to",
        validation_data["date"].max()
    )

    print("\nTraining rows:", f"{len(train_data):,}")
    print(
        "Validation rows:",
        f"{len(validation_data):,}"
    )

    if train_data.empty or validation_data.empty:
        raise ValueError(
            "Training or validation dataset is empty."
        )

    return train_data, validation_data


# ==================================================
# MODEL PIPELINE
# ==================================================

def create_pipeline(model):

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1
                        )
                    )
                ]),
                CATEGORICAL_FEATURES
            )
        ],
        remainder="passthrough"
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("model", model)
    ])

    return pipeline


# ==================================================
# TRAIN AND EVALUATE
# ==================================================

def main():

    data = load_training_sample()

    train_data, validation_data = split_data(data)

    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET]

    X_valid = validation_data[FEATURE_COLUMNS]
    y_valid = validation_data[TARGET]

    # ----------------------------------------------
    # BASELINE MODEL
    # ----------------------------------------------

    print("\n" + "=" * 60)
    print("BASELINE MODEL")
    print("=" * 60)

    baseline_predictions = (
        validation_data["rolling_mean_7"]
        .to_numpy()
    )

    results = []

    baseline_metrics = calculate_metrics(
        y_valid.to_numpy(),
        baseline_predictions
    )

    baseline_metrics["Model"] = "7-Day Average Baseline"

    results.append(baseline_metrics)

    print(baseline_metrics)

    # ----------------------------------------------
    # MACHINE LEARNING MODELS
    # ----------------------------------------------

    models = {

        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=60,
            max_depth=12,
            min_samples_leaf=5,
            max_features=0.8,
            n_jobs=2,
            random_state=42
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=100,
            max_leaf_nodes=15,
            learning_rate=0.08,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=42
        )
    }

    best_rmse = float("inf")
    best_model_name = None

    for model_name, model in models.items():

        print("\n" + "=" * 60)
        print("TRAINING:", model_name)
        print("=" * 60)

        pipeline = create_pipeline(model)

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(X_valid)

        # Sales cannot be negative.
        predictions = np.maximum(
            predictions, 0
        )

        metrics = calculate_metrics(
            y_valid.to_numpy(),
            predictions
        )

        metrics["Model"] = model_name

        results.append(metrics)

        print("\nEvaluation results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

        model_filename = (
            model_name.lower()
            .replace(" ", "_")
            + ".joblib"
        )

        model_path = os.path.join(
            MODEL_DIR,
            model_filename
        )

        joblib.dump(
            pipeline,
            model_path
        )

        print("\nSaved model:", model_path)

        if metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_model_name = model_name

        del pipeline
        del predictions
        gc.collect()

    # ----------------------------------------------
    # SAVE RESULTS
    # ----------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "Model",
            "MAE",
            "RMSE",
            "MAPE_nonzero_percent",
            "WAPE_percent"
        ]
    ]

    results_path = os.path.join(
        RESULT_DIR,
        "model_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    best_model_info = {
        "best_model": best_model_name,
        "best_rmse": best_rmse,
        "validation_days": VALIDATION_DAYS,
        "sample_limit": SAMPLE_LIMIT
    }

    with open(
        os.path.join(
            MODEL_DIR,
            "best_model.json"
        ),
        "w"
    ) as file:
        json.dump(
            best_model_info,
            file,
            indent=4
        )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(results_df.to_string(index=False))

    print("\nBest model:", best_model_name)
    print("Best RMSE:", best_rmse)

    print("\nResults saved to:", results_path)
    print("Models saved to:", MODEL_DIR)

    print("\nTraining completed successfully!")

    del data
    del train_data
    del validation_data
    gc.collect()


if __name__ == "__main__":
    main()

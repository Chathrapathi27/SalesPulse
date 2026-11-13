import gc
import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

FEATURE_DIR = os.path.join(BASE_DIR, "data/processed/features")
RESULT_DIR = os.path.join(BASE_DIR, "data/processed/model_results")

os.makedirs(RESULT_DIR, exist_ok=True)

SAMPLE_MODULO = 30490
SAMPLE_LIMIT = 100
FORECAST_DAYS = 28

CATEGORICAL_FEATURES = [
    "item_id", "dept_id", "cat_id", "store_id", "state_id"
]

NUMERIC_FEATURES = [
    "wday", "month", "year",
    "snap_CA", "snap_TX", "snap_WI",
    "day_of_week", "is_weekend",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_28",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

READ_COLUMNS = [
    "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "date", "sales", "wday", "month", "year",
    "snap_CA", "snap_TX", "snap_WI",
    "day_of_week", "is_weekend",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_28",
]


def load_sample():
    files = sorted(glob.glob(os.path.join(FEATURE_DIR, "features_*.parquet")))

    if not files:
        raise FileNotFoundError(f"No feature files found: {FEATURE_DIR}")

    parts = []

    print(f"Reading {len(files)} feature files...")

    for file_path in tqdm(files):
        df = pd.read_parquet(file_path, columns=READ_COLUMNS)

        selected_mask = (
            pd.util.hash_pandas_object(df["id"], index=False)
            % SAMPLE_MODULO
        ) < SAMPLE_LIMIT

        selected = df.loc[selected_mask].copy()

        if not selected.empty:
            parts.append(selected)

        del df, selected
        gc.collect()

    if not parts:
        raise ValueError("No product series selected.")

    data = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    data["date"] = pd.to_datetime(data["date"])
    data["sales"] = pd.to_numeric(data["sales"], errors="raise")

    data = data.sort_values(["id", "date"]).reset_index(drop=True)

    print("\nSample information")
    print("Rows:", f"{len(data):,}")
    print("Product series:", data["id"].nunique())
    print("Date range:", data["date"].min(), "to", data["date"].max())

    return data


def create_pipeline(model):
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ])

    preprocessing = ColumnTransformer([
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ("numeric", "passthrough", NUMERIC_FEATURES),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", model),
    ])


def create_models():
    return {
        "Linear Regression": create_pipeline(LinearRegression()),

        "Random Forest": create_pipeline(
            RandomForestRegressor(
                n_estimators=60,
                max_depth=12,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=2,
            )
        ),

        "HistGradientBoosting": create_pipeline(
            HistGradientBoostingRegressor(
                max_iter=100,
                max_leaf_nodes=15,
                learning_rate=0.08,
                random_state=42,
            )
        ),
    }


def build_forecast_features(calendar_row, history):
    features = {
        column: calendar_row[column]
        for column in CATEGORICAL_FEATURES + [
            "wday", "month", "year",
            "snap_CA", "snap_TX", "snap_WI",
            "day_of_week", "is_weekend",
        ]
    }

    features.update({
        "lag_1": history[-1],
        "lag_7": history[-7],
        "lag_14": history[-14],
        "lag_28": history[-28],
        "rolling_mean_7": float(np.mean(history[-7:])),
        "rolling_mean_28": float(np.mean(history[-28:])),
    })

    return features


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    nonzero = actual != 0
    mape = (
        np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero])
                / actual[nonzero]
            )
        ) * 100
        if nonzero.any()
        else np.nan
    )

    total_actual = np.abs(actual).sum()
    wape = (
        np.abs(actual - predicted).sum() / total_actual * 100
        if total_actual > 0
        else np.nan
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE_nonzero_percent": float(mape),
        "WAPE_percent": float(wape),
    }


def main():
    print("=" * 65)
    print("SALESPULSE - RECURSIVE 28-DAY FORECAST EVALUATION")
    print("=" * 65)

    data = load_sample()

    final_date = data["date"].max()
    forecast_start = final_date - pd.Timedelta(days=FORECAST_DAYS - 1)

    training_data = data[data["date"] < forecast_start].copy()
    forecast_data = data[data["date"] >= forecast_start].copy()

    expected_dates = pd.date_range(
        forecast_start, final_date, freq="D"
    )

    actual_dates = pd.DatetimeIndex(
        sorted(forecast_data["date"].unique())
    )

    if not actual_dates.equals(expected_dates):
        raise ValueError("The holdout period does not contain 28 consecutive dates.")

    print("\nTraining period:")
    print(training_data["date"].min(), "to", training_data["date"].max())

    print("\nHoldout period:")
    print(forecast_start, "to", final_date)

    print("\nTraining rows:", f"{len(training_data):,}")
    print("Holdout rows:", f"{len(forecast_data):,}")

    models = create_models()

    X_train = training_data[FEATURE_COLUMNS]
    y_train = training_data["sales"]

    print("\nTraining models...")

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        print(f"{name} completed.")

    del X_train, y_train, training_data
    gc.collect()

    # Initialize an independent history for every model and product.
    initial_histories = {}

    for product_id, group in data[
        data["date"] < forecast_start
    ].groupby("id", sort=False):

        history = group.sort_values("date")["sales"].astype(float).tolist()

        if len(history) >= 28:
            initial_histories[product_id] = history

    histories = {
        model_name: {
            product_id: history.copy()
            for product_id, history in initial_histories.items()
        }
        for model_name in models
    }

    # Baseline has its own independent recursive history too.
    histories["7-Day Average Baseline"] = {
        product_id: history.copy()
        for product_id, history in initial_histories.items()
    }

    del initial_histories, data
    gc.collect()

    predictions_output = []
    daily_metrics_output = []

    forecast_data = forecast_data.sort_values(
        ["date", "id"]
    )

    print("\nGenerating independent recursive forecasts...")

    for day_number, forecast_date in enumerate(expected_dates, start=1):
        daily = forecast_data[
            forecast_data["date"] == forecast_date
        ]

        daily = daily.set_index("id", drop=False)

        # Keep only products with sufficient training history.
        product_ids = [
            product_id
            for product_id in daily.index
            if product_id in histories["Random Forest"]
        ]

        if not product_ids:
            print(f"No eligible products on {forecast_date.date()}.")
            continue

        daily_actuals = []
        daily_predictions = {
            name: [] for name in histories
        }

        for product_id in product_ids:
            row = daily.loc[product_id]

            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            calendar_row = row.to_dict()
            actual = float(row["sales"])
            daily_actuals.append(actual)

            # Each strategy builds features from its own history.
            for model_name in histories:
                history = histories[model_name][product_id]

                if model_name == "7-Day Average Baseline":
                    prediction = float(np.mean(history[-7:]))
                else:
                    features = build_forecast_features(
                        calendar_row, history
                    )

                    X_future = pd.DataFrame(
                        [features],
                        columns=FEATURE_COLUMNS,
                    )

                    prediction = float(
                        models[model_name].predict(X_future)[0]
                    )

                prediction = max(0.0, prediction)
                daily_predictions[model_name].append(prediction)

                predictions_output.append({
                    "id": product_id,
                    "date": forecast_date.strftime("%Y-%m-%d"),
                    "forecast_day": day_number,
                    "actual_sales": actual,
                    "model": model_name,
                    "predicted_sales": prediction,
                })

                # Recursive update uses this model's own prediction.
                history.append(prediction)

        for model_name, values in daily_predictions.items():
            metrics = calculate_metrics(daily_actuals, values)

            daily_metrics_output.append({
                "forecast_day": day_number,
                "date": forecast_date.strftime("%Y-%m-%d"),
                "model": model_name,
                **metrics,
            })

        print(
            f"Day {day_number:02d}/28 "
            f"| {forecast_date.date()} "
            f"| {len(product_ids)} product series"
        )

    predictions_df = pd.DataFrame(predictions_output)
    daily_metrics_df = pd.DataFrame(daily_metrics_output)

    summary_rows = []

    for model_name, group in predictions_df.groupby("model"):
        metrics = calculate_metrics(
            group["actual_sales"],
            group["predicted_sales"],
        )

        summary_rows.append({
            "model": model_name,
            **metrics,
            "forecast_days": group["date"].nunique(),
            "product_series": group["id"].nunique(),
            "forecast_records": len(group),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("RMSE")

    predictions_file = os.path.join(
        RESULT_DIR, "recursive_28day_predictions.csv"
    )
    summary_file = os.path.join(
        RESULT_DIR, "recursive_28day_summary.csv"
    )
    daily_file = os.path.join(
        RESULT_DIR, "recursive_28day_daily_metrics.csv"
    )

    predictions_df.to_csv(predictions_file, index=False)
    summary_df.to_csv(summary_file, index=False)
    daily_metrics_df.to_csv(daily_file, index=False)

    print("\n" + "=" * 65)
    print("28-DAY RECURSIVE EVALUATION RESULTS")
    print("=" * 65)
    print(summary_df.to_string(index=False))

    print("\nSaved evaluation files:")
    print(predictions_file)
    print(summary_file)
    print(daily_file)


if __name__ == "__main__":
    main()

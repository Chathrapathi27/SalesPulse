
import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

# ============================================================
# SALESPULSE - MLFLOW EXPERIMENT TRACKING
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

RESULT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "model_results",
    "model_comparison.csv"
)

MLFLOW_DB = os.path.join(BASE_DIR, "mlflow.db")

EXPERIMENT_NAME = "SalesPulse_Model_Experiments"

TRACKING_URI = (
    "sqlite:///"
    + MLFLOW_DB.replace("\\", "/")
)

# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

mlflow.set_tracking_uri(TRACKING_URI)

mlflow.set_experiment(EXPERIMENT_NAME)

print("MLflow tracking database:", MLFLOW_DB)
print("MLflow experiment:", EXPERIMENT_NAME)

# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_FILES = {
    "Linear Regression": "linear_regression.joblib",
    "Random Forest": "random_forest.joblib",
    "HistGradientBoosting": "histgradientboosting.joblib"
}

METRIC_COLUMNS = [
    "MAE",
    "RMSE",
    "MAPE_nonzero_percent",
    "WAPE_percent"
]

# Explicitly trusted types for models created by
# this project's own training pipeline.
TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
]

# ============================================================
# HELPER: LOG METRICS
# ============================================================

def log_metrics(metrics):

    for metric_name in METRIC_COLUMNS:

        value = metrics.get(metric_name)

        if pd.notna(value) and np.isfinite(float(value)):

            mlflow.log_metric(
                metric_name,
                float(value)
            )

# ============================================================
# HELPER: CHECK EXISTING RUN
# ============================================================

def run_already_logged(model_name, run_type="trained_model"):

    experiment = mlflow.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        return False

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.model_name = '{model_name}' "
            f"and tags.run_type = '{run_type}'"
        )
    )

    if runs.empty:
        return False

    # Do not repeat a successfully completed run.
    return any(
        runs["status"] == "FINISHED"
    )

# ============================================================
# TRACK TRAINED MODELS
# ============================================================

def track_model(model_name, filename, metrics):

    model_path = os.path.join(
        MODEL_DIR,
        filename
    )

    if not os.path.exists(model_path):

        print(f"\nModel file not found: {model_path}")
        return False

    if run_already_logged(model_name):

        print(
            f"\nAlready tracked successfully: {model_name}"
        )

        return True

    print("\n" + "=" * 60)
    print("TRACKING MODEL:", model_name)
    print("=" * 60)

    try:

        model = joblib.load(model_path)

        with mlflow.start_run(
            run_name=model_name
        ):

            # Run identification
            mlflow.set_tag(
                "model_name",
                model_name
            )

            mlflow.set_tag(
                "run_type",
                "trained_model"
            )

            mlflow.set_tag(
                "project",
                "SalesPulse"
            )

            # Parameters
            mlflow.log_param(
                "training_dataset",
                "M5 Forecasting Accuracy"
            )

            mlflow.log_param(
                "validation_days",
                28
            )

            mlflow.log_param(
                "sample_limit",
                100
            )

            mlflow.log_param(
                "model_file",
                filename
            )

            # Evaluation metrics
            log_metrics(metrics)

            # Log the Scikit-learn model
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                skops_trusted_types=TRUSTED_TYPES
            )

            # Also log the original joblib file
            mlflow.log_artifact(
                model_path,
                artifact_path="joblib_model"
            )

        print(
            f"Successfully tracked: {model_name}"
        )

        del model

        return True

    except Exception as error:

        print(
            f"\nERROR tracking {model_name}: {error}"
        )

        return False

# ============================================================
# TRACK BASELINE
# ============================================================

def track_baseline(metrics):

    model_name = "7-Day Average Baseline"

    if run_already_logged(
        model_name,
        run_type="baseline"
    ):

        print("\nBaseline already tracked successfully.")
        return True

    print("\n" + "=" * 60)
    print("TRACKING BASELINE MODEL")
    print("=" * 60)

    try:

        with mlflow.start_run(
            run_name=model_name
        ):

            mlflow.set_tag(
                "model_name",
                model_name
            )

            mlflow.set_tag(
                "run_type",
                "baseline"
            )

            mlflow.set_tag(
                "project",
                "SalesPulse"
            )

            mlflow.log_param(
                "model_type",
                "7-day rolling average"
            )

            mlflow.log_param(
                "training_dataset",
                "M5 Forecasting Accuracy"
            )

            mlflow.log_param(
                "validation_days",
                28
            )

            log_metrics(metrics)

        print("Successfully tracked baseline.")

        return True

    except Exception as error:

        print("\nERROR tracking baseline:", error)

        return False

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("SALESPULSE - MLFLOW EXPERIMENT TRACKING")
    print("=" * 60)

    # Check evaluation file
    if not os.path.exists(RESULT_FILE):

        raise FileNotFoundError(
            f"Model comparison file not found: {RESULT_FILE}"
        )

    # Load evaluation results
    results = pd.read_csv(RESULT_FILE)

    if "Model" not in results.columns:

        raise ValueError(
            "The results CSV does not contain a Model column."
        )

    successful_models = 0
    failed_models = 0

    # Track machine-learning models
    for model_name, filename in MODEL_FILES.items():

        row = results[
            results["Model"] == model_name
        ]

        if row.empty:

            print(
                f"\nNo evaluation results found for {model_name}"
            )

            failed_models += 1
            continue

        metrics = row.iloc[0].to_dict()

        success = track_model(
            model_name,
            filename,
            metrics
        )

        if success:
            successful_models += 1
        else:
            failed_models += 1

    # Track baseline
    baseline = results[
        results["Model"] == "7-Day Average Baseline"
    ]

    if not baseline.empty:

        baseline_metrics = baseline.iloc[0].to_dict()

        if track_baseline(baseline_metrics):
            successful_models += 1
        else:
            failed_models += 1

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 60)
    print("MLFLOW TRACKING SUMMARY")
    print("=" * 60)

    print("Successfully tracked:", successful_models)
    print("Failed models:", failed_models)

    print("Experiment:", EXPERIMENT_NAME)
    print("Tracking database:", MLFLOW_DB)

    if failed_models == 0:

        print("\nAll models tracked successfully!")

    else:

        print(
            "\nSome models were not tracked. "
            "Review the errors above."
        )

    print("\nMLflow tracking process completed.")


if __name__ == "__main__":
    main()

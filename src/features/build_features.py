
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

# --------------------------------------------------
# SALESPULSE - FEATURE ENGINEERING
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

INPUT_DIR = os.path.join(
    BASE_DIR, "data", "processed", "m5_sales"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "processed", "features"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

LAG_DAYS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 28]

# --------------------------------------------------
# FEATURE ENGINEERING FUNCTION
# --------------------------------------------------

def create_features(df):

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    # Sort each product-store series chronologically
    df = df.sort_values(
        ["id", "date"]
    ).reset_index(drop=True)

    # Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype("int8")

    # Existing calendar columns are retained:
    # month, year, wday, SNAP and event information.

    # Historical sales features
    grouped_sales = df.groupby(
        "id", sort=False
    )["sales"]

    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = grouped_sales.shift(lag)

    # Rolling averages must use previous days only
    for window in ROLLING_WINDOWS:

        df[f"rolling_mean_{window}"] = (
            df.groupby("id", sort=False)["sales"]
            .transform(
                lambda series: (
                    series.shift(1)
                    .rolling(
                        window=window,
                        min_periods=window
                    )
                    .mean()
                )
            )
        )

    # Remove rows without enough historical observations
    required_features = (
        [f"lag_{lag}" for lag in LAG_DAYS]
        + [
            f"rolling_mean_{window}"
            for window in ROLLING_WINDOWS
        ]
    )

    df = df.dropna(
        subset=required_features
    ).reset_index(drop=True)

    return df


# --------------------------------------------------
# PROCESS PARQUET FILES
# --------------------------------------------------

def main():

    files = sorted(
        glob.glob(
            os.path.join(INPUT_DIR, "part_*.parquet")
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in {INPUT_DIR}"
        )

    print("=" * 60)
    print("SALESPULSE - FEATURE ENGINEERING")
    print("=" * 60)

    print(f"Input files: {len(files)}")

    # Each file contains different product-store series.
    # Therefore, lags can safely be calculated per file.
    total_rows = 0

    for index, file_path in enumerate(
        tqdm(files, desc="Building features"), start=1
    ):

        df = pd.read_parquet(file_path)

        featured_df = create_features(df)

        output_path = os.path.join(
            OUTPUT_DIR,
            f"features_{index:03d}.parquet"
        )

        featured_df.to_parquet(
            output_path,
            index=False
        )

        total_rows += len(featured_df)

        del df
        del featured_df

    print("\nFeature engineering completed!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Output files: {len(files)}")
    print(f"Total feature rows: {total_rows:,}")


if __name__ == "__main__":
    main()

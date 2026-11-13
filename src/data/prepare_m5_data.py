
import os
import json
import shutil
import pandas as pd
from tqdm import tqdm

# ==========================================
# SALESPULSE - FULL M5 DATA TRANSFORMATION
# ==========================================

RAW_DIR = "data/raw"
OUTPUT_DIR = "data/processed/m5_sales"

INPUT_FILE = os.path.join(
    RAW_DIR, "sales_train_evaluation.csv"
)

CALENDAR_FILE = os.path.join(
    RAW_DIR, "calendar.csv"
)

CHUNK_SIZE = 250

# ==========================================
# STEP 1: VALIDATE INPUT FILES
# ==========================================

print("\nSALESPULSE: M5 DATA TRANSFORMATION")
print("=" * 60)

for path in [INPUT_FILE, CALENDAR_FILE]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

# ==========================================
# STEP 2: LOAD CALENDAR
# ==========================================

print("\nLoading calendar...")

calendar = pd.read_csv(CALENDAR_FILE)

calendar["date"] = pd.to_datetime(calendar["date"])

# Keep useful calendar columns
calendar_columns = [
    "d",
    "date",
    "wm_yr_wk",
    "weekday",
    "wday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI"
]

calendar_columns = [
    c for c in calendar_columns
    if c in calendar.columns
]

calendar = calendar[calendar_columns]

print("Calendar loaded:", calendar.shape)

# ==========================================
# STEP 3: READ SALES COLUMN INFORMATION
# ==========================================

print("\nInspecting sales columns...")

header = pd.read_csv(INPUT_FILE, nrows=0)

metadata_columns = [
    c for c in header.columns
    if not c.startswith("d_")
]

daily_columns = [
    c for c in header.columns
    if c.startswith("d_")
]

print("Product-store combinations expected: 30,490")
print("Number of daily columns:", len(daily_columns))

# ==========================================
# STEP 4: PREPARE OUTPUT DIRECTORY
# ==========================================

# Remove old output to prevent duplicate chunks
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# STEP 5: PROCESS DATA IN CHUNKS
# ==========================================

print("\nStarting full dataset transformation...")

reader = pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE
)

total_rows = 0
total_sales = 0
chunk_number = 0

for chunk in tqdm(reader, desc="Processing sales"):

    # Convert wide data into long format
    long_data = chunk.melt(
        id_vars=metadata_columns,
        value_vars=daily_columns,
        var_name="d",
        value_name="sales"
    )

    # Merge with calendar
    long_data = long_data.merge(
        calendar,
        on="d",
        how="left",
        validate="many_to_one"
    )

    # Convert sales to numeric
    long_data["sales"] = pd.to_numeric(
        long_data["sales"],
        errors="coerce"
    )

    # Convert metadata to category types
    for column in [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id"
    ]:
        if column in long_data.columns:
            long_data[column] = (
                long_data[column].astype("category")
            )

    # Arrange columns
    preferred_columns = [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "d",
        "date",
        "sales",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI"
    ]

    output_columns = [
        c for c in preferred_columns
        if c in long_data.columns
    ]

    long_data = long_data[output_columns]

    # Save individual Parquet chunk
    output_file = os.path.join(
        OUTPUT_DIR,
        f"part_{chunk_number:04d}.parquet"
    )

    long_data.to_parquet(
        output_file,
        index=False,
        compression="snappy"
    )

    total_rows += len(long_data)

    total_sales += long_data["sales"].sum()

    chunk_number += 1

    del long_data
    del chunk

# ==========================================
# STEP 6: GENERATE DATASET SUMMARY
# ==========================================

summary = {
    "dataset": "M5 Forecasting Accuracy",
    "source_file": "sales_train_evaluation.csv",
    "product_store_combinations": 30490,
    "daily_observations": len(daily_columns),
    "total_records": total_rows,
    "total_sales": float(total_sales),
    "chunks_created": chunk_number,
    "output_format": "Parquet"
}

summary_path = os.path.join(
    OUTPUT_DIR, "dataset_summary.json"
)

with open(summary_path, "w") as file:
    json.dump(summary, file, indent=4)

# ==========================================
# STEP 7: DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 60)
print("TRANSFORMATION COMPLETED SUCCESSFULLY")
print("=" * 60)

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nOutput directory:", OUTPUT_DIR)

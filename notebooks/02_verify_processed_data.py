
import os
import glob
import json
import pandas as pd

DATA_DIR = "data/processed/m5_sales"

print("=" * 55)
print("SALESPULSE - PROCESSED DATA VERIFICATION")
print("=" * 55)

# Read summary
summary_path = os.path.join(
    DATA_DIR, "dataset_summary.json"
)

with open(summary_path, "r") as file:
    summary = json.load(file)

print("\nDataset Summary:")
for key, value in summary.items():
    print(f"{key}: {value}")

# Find Parquet chunks
files = sorted(
    glob.glob(os.path.join(DATA_DIR, "part_*.parquet"))
)

print("\nNumber of Parquet files:", len(files))

# Load first chunk for inspection
df = pd.read_parquet(files[0])

print("\nFirst chunk shape:", df.shape)

print("\nFirst 10 records:")
print(df.head(10).to_string(index=False))

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())

print("\nSales statistics:")
print(df["sales"].describe())

print("\nVerification completed!")

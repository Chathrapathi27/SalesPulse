
import os
import glob
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

FEATURE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "features"
)

files = sorted(
    glob.glob(
        os.path.join(FEATURE_DIR, "features_*.parquet")
    )
)

print("=" * 60)
print("SALESPULSE - FEATURE VERIFICATION")
print("=" * 60)

print("\nNumber of feature files:", len(files))

if not files:
    raise FileNotFoundError(
        "No feature files found. Run build_features.py first."
    )

df = pd.read_parquet(files[0])

print("\nFirst feature file shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 records:")
print(df.head(10).to_string(index=False))

feature_columns = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28"
]

print("\nMissing values in historical features:")
print(df[feature_columns].isnull().sum())

print("\nFeature statistics:")
print(df[feature_columns].describe())

print("\nTarget sales statistics:")
print(df["sales"].describe())

print("\nVerification completed!")

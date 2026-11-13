
import os
import pandas as pd

# ==========================================
# SALESPULSE - M5 DATASET INSPECTION
# ==========================================

DATA_DIR = "data/raw"

required_files = [
    "calendar.csv",
    "sales_train_validation.csv",
    "sales_train_evaluation.csv",
    "sell_prices.csv",
    "sample_submission.csv"
]

print("=" * 65)
print("          SALESPULSE - M5 DATASET INSPECTION")
print("=" * 65)

# 1. Check whether all files exist
print("\n1. CHECKING DATASET FILES")

for filename in required_files:
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"[FOUND] {filename} ({size_mb:.2f} MB)")
    else:
        print(f"[MISSING] {filename}")

missing = [
    f for f in required_files
    if not os.path.exists(os.path.join(DATA_DIR, f))
]

if missing:
    raise FileNotFoundError(
        f"Missing dataset files: {missing}"
    )

# 2. Inspect sales dataset structure
print("\n2. SALES DATASET INFORMATION")

sales_path = os.path.join(
    DATA_DIR, "sales_train_evaluation.csv"
)

sales = pd.read_csv(sales_path, nrows=5)

print("First five records:")
print(sales.head())

print("\nColumns:")
print(sales.columns.tolist()[:10], "...")

print("\nNumber of metadata columns:",
      len([c for c in sales.columns if not c.startswith("d_")]))

print("Number of daily sales columns:",
      len([c for c in sales.columns if c.startswith("d_")]))

# 3. Inspect calendar
print("\n3. CALENDAR DATASET")

calendar = pd.read_csv(
    os.path.join(DATA_DIR, "calendar.csv")
)

print("Shape:", calendar.shape)
print(calendar.head())

# 4. Inspect prices
print("\n4. SELL PRICES DATASET")

prices = pd.read_csv(
    os.path.join(DATA_DIR, "sell_prices.csv"),
    nrows=5
)

print(prices.head())
print("Columns:", prices.columns.tolist())

# 5. Inspect submission
print("\n5. SAMPLE SUBMISSION DATASET")

submission = pd.read_csv(
    os.path.join(DATA_DIR, "sample_submission.csv"),
    nrows=5
)

print(submission.head())

print("\n" + "=" * 65)
print("DATASET INSPECTION COMPLETED SUCCESSFULLY")
print("=" * 65)

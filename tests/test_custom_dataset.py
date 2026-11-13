
import pandas as pd
import pytest

from src.prediction import custom_dataset


def make_valid_data(days=35):
    """Create a valid sample dataset for testing."""

    dates = pd.date_range(
        start="2025-01-01",
        periods=days,
        freq="D"
    )

    return pd.DataFrame({
        "id": ["PRODUCT_001"] * days,
        "date": dates,
        "sales": [10.0 + (i % 5) for i in range(days)]
    })


def test_invalid_date_is_rejected():
    data = make_valid_data()
    data["date"] = data["date"].astype("object")
    data.loc[0, "date"] = "not-a-date"

    with pytest.raises(ValueError, match="Invalid or missing dates"):
        custom_dataset.validate_dataset(data)


def test_non_numeric_sales_are_rejected():
    data = make_valid_data()
    data["sales"] = data["sales"].astype("object")
    data.loc[0, "sales"] = "unknown"

    with pytest.raises(ValueError, match="valid numeric values"):
        custom_dataset.validate_dataset(data)


def test_negative_sales_are_rejected():
    data = make_valid_data()
    data.loc[0, "sales"] = -5

    with pytest.raises(ValueError, match="Negative sales values"):
        custom_dataset.validate_dataset(data)


def test_duplicate_product_date_rows_are_rejected():
    data = make_valid_data()

    duplicate = data.iloc[[0]].copy()
    data = pd.concat(
        [data, duplicate],
        ignore_index=True
    )

    with pytest.raises(
        ValueError,
        match="Duplicate product/date records"
    ):
        custom_dataset.validate_dataset(data)

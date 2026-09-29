"""
Unit tests for SmartHouse AI EDA module.
"""
import pytest
import pandas as pd
import numpy as np
from src.data.eda import (
    compute_dataset_overview,
    compute_target_statistics,
    calculate_price_per_sqft,
    compute_numerical_summaries,
    compute_categorical_summaries,
    compute_correlation_matrix,
    compute_location_statistics,
    compute_outlier_summary
)


@pytest.fixture
def mock_df():
    return pd.DataFrame({
        "location": ["Whitefield", "Whitefield", "Koramangala", "Koramangala", "HSR Layout", "Other"],
        "total_sqft": [1000.0, 2000.0, 1500.0, 1800.0, 1200.0, 900.0],
        "bhk": [2, 3, 3, 3, 2, 2],
        "bath": [2, 3, 3, 3, 2, 1],
        "balcony": [1, 2, 2, 2, 1, 1],
        "area_type": ["Super built-up Area", "Super built-up Area", "Plot Area", "Plot Area", "Built-up Area", "Super built-up Area"],
        "availability_status": ["Ready To Move", "Under Construction", "Ready To Move", "Ready To Move", "Under Construction", "Ready To Move"],
        "price": [50.0, 120.0, 180.0, 220.0, 90.0, 40.0]
    })


def test_compute_dataset_overview(mock_df):
    res = compute_dataset_overview(mock_df)
    assert res["total_rows"] == 6
    assert res["total_columns"] == 8
    assert "total_sqft" in res["numerical_columns"]
    assert "location" in res["categorical_columns"]


def test_compute_target_statistics(mock_df):
    res = compute_target_statistics(mock_df, target_col="price")
    assert res["target_name"] == "price"
    assert res["count"] == 6
    assert res["min"] == 40.0
    assert res["max"] == 220.0
    assert res["mean"] > 0
    assert "skewness" in res
    assert "log_skewness" in res


def test_calculate_price_per_sqft(mock_df):
    psqft = calculate_price_per_sqft(mock_df)
    assert len(psqft) == len(mock_df)
    # Row 0: price = 50 Lakhs = 5,000,000 / 1000 sqft = 5000 INR/sqft
    assert psqft.iloc[0] == 5000.0


def test_compute_numerical_summaries(mock_df):
    res = compute_numerical_summaries(mock_df)
    assert "total_sqft" in res
    assert "bhk" in res
    assert res["total_sqft"]["mean"] == 1400.0


def test_compute_categorical_summaries(mock_df):
    res = compute_categorical_summaries(mock_df)
    assert "location" in res
    assert "area_type" in res
    assert res["location"]["unique_count"] == 4


def test_compute_correlation_matrix(mock_df):
    corr = compute_correlation_matrix(mock_df)
    assert "price" in corr.columns
    assert "total_sqft" in corr.columns
    assert corr.loc["price", "price"] == 1.0


def test_compute_location_statistics(mock_df):
    res = compute_location_statistics(mock_df, min_listings=2)
    assert res["total_unique_locations"] == 4
    # Whitefield (2) and Koramangala (2) have >= 2 listings
    assert res["qualified_locations_count"] == 2


def test_compute_outlier_summary(mock_df):
    res = compute_outlier_summary(mock_df)
    assert "price" in res
    assert "lower_fence" in res["price"]
    assert "upper_fence" in res["price"]


"""
Unit tests for SmartHouse AI data validation module.
"""
import pytest
import pandas as pd
import numpy as np
from src.data.validation import DataValidator


@pytest.fixture
def mock_clean_df():
    return pd.DataFrame({
        "location": ["Whitefield", "Koramangala", "HSR Layout", "Electronic City"],
        "total_sqft": [1200.0, 1600.0, 2000.0, 950.0],
        "bhk": [2, 3, 3, 2],
        "bath": [2, 3, 3, 2],
        "balcony": [1, 2, 2, 1],
        "price": [70.0, 130.0, 175.0, 48.0]
    })


@pytest.fixture
def mock_dirty_df():
    return pd.DataFrame({
        "location": ["Whitefield", None, "HSR Layout", "HSR Layout"],
        "total_sqft": [-500.0, 1600.0, 0.0, 1000.0],
        "bhk": [2, -1, 3, 3],
        "bath": [8, 3, 3, 3],  # 8 bath for 2 BHK -> impossible
        "balcony": [1, np.nan, 2, 2],
        "price": [-10.0, 130.0, 0.0, 90.0]  # negative/zero prices -> impossible
    })


def test_validate_schema(mock_clean_df):
    validator = DataValidator(required_columns=["location", "total_sqft", "price"])
    res = validator.validate_schema(mock_clean_df)
    assert res["valid"] is True
    assert len(res["missing_required_columns"]) == 0

    bad_validator = DataValidator(required_columns=["location", "non_existent"])
    res_bad = bad_validator.validate_schema(mock_clean_df)
    assert res_bad["valid"] is False
    assert "non_existent" in res_bad["missing_required_columns"]


def test_check_missing_values(mock_dirty_df):
    validator = DataValidator()
    res = validator.check_missing_values(mock_dirty_df, critical_columns=["location", "price"])
    assert res["total_missing_cells"] > 0
    assert "location" in res["critical_missing"]


def test_check_duplicates(mock_clean_df):
    validator = DataValidator()
    res = validator.check_duplicates(mock_clean_df)
    assert res["duplicate_count"] == 0

    dup_df = pd.concat([mock_clean_df, mock_clean_df.iloc[[0]]], ignore_index=True)
    res_dup = validator.check_duplicates(dup_df)
    assert res_dup["duplicate_count"] == 1


def test_check_impossible_values(mock_dirty_df):
    validator = DataValidator()
    res = validator.check_impossible_values(mock_dirty_df)
    assert res["impossible_count"] > 0
    breakdown = res["anomalies_breakdown"]
    assert "non_positive_price_count" in breakdown
    assert "excessive_bathrooms_count" in breakdown
    assert "non_positive_sqft_count" in breakdown


def test_detect_outliers_iqr():
    df = pd.DataFrame({
        "price": [10.0, 12.0, 11.0, 13.0, 12.5, 11.8, 500.0]  # 500 is extreme outlier
    })
    validator = DataValidator()
    res = validator.detect_outliers_iqr(df, columns=["price"])
    assert "price" in res
    assert res["price"]["outlier_count"] == 1


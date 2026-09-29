"""
Unit tests for SmartHouse AI feature engineering and leakage safeguards.
"""
import pytest
import numpy as np
import pandas as pd
from src.features.engineering import (
    FeatureEngineeringTransformer,
    validate_no_target_leakage,
    transform_target,
    inverse_transform_target,
    LEAKAGE_FEATURE_BLACKLIST
)


@pytest.fixture
def sample_feature_df():
    return pd.DataFrame({
        "location": ["Whitefield", "Koramangala", "Indiranagar"],
        "total_sqft": [1200.0, 1800.0, 2400.0],
        "bath": [2.0, 3.0, 3.0],
        "balcony": [1.0, 2.0, 2.0],
        "bhk": [2.0, 3.0, 4.0],
        "area_type": ["Super built-up Area", "Plot Area", "Built-up Area"],
        "availability_status": ["Ready To Move", "Under Construction", "Ready To Move"]
    })


def test_feature_engineering_transformer(sample_feature_df):
    transformer = FeatureEngineeringTransformer()
    transformed_df = transformer.fit_transform(sample_feature_df)

    assert "total_rooms" in transformed_df.columns
    assert "sqft_per_bhk" in transformed_df.columns
    assert "bath_per_bhk" in transformed_df.columns
    assert "sqft_per_room" in transformed_df.columns
    assert "bhk_bath_diff" in transformed_df.columns

    # Row 0: 2 BHK, 2 Bath -> total_rooms = 4.0, sqft_per_bhk = 600.0, bath_per_bhk = 1.0
    assert transformed_df.loc[0, "total_rooms"] == 4.0
    assert transformed_df.loc[0, "sqft_per_bhk"] == 600.0
    assert transformed_df.loc[0, "bath_per_bhk"] == 1.0
    assert transformed_df.loc[0, "bhk_bath_diff"] == 0.0


def test_division_by_zero_safety():
    edge_df = pd.DataFrame({
        "total_sqft": [1000.0],
        "bath": [0.0],
        "balcony": [0.0],
        "bhk": [0.0],
        "location": ["Whitefield"],
        "area_type": ["Super built-up Area"],
        "availability_status": ["Ready To Move"]
    })
    transformer = FeatureEngineeringTransformer()
    transformed = transformer.fit_transform(edge_df)

    assert not np.isnan(transformed["sqft_per_bhk"].iloc[0])
    assert not np.isinf(transformed["sqft_per_bhk"].iloc[0])
    assert not np.isnan(transformed["sqft_per_room"].iloc[0])
    assert not np.isinf(transformed["sqft_per_room"].iloc[0])


def test_leakage_exclusion_clean_features(sample_feature_df):
    # Should not raise any error
    validate_no_target_leakage(sample_feature_df.columns)


def test_leakage_exclusion_blacklisted_features():
    for blacklisted_col in LEAKAGE_FEATURE_BLACKLIST:
        dirty_cols = ["total_sqft", "bhk", blacklisted_col]
        with pytest.raises(ValueError, match="CRITICAL DATA LEAKAGE DETECTED"):
            validate_no_target_leakage(dirty_cols)


def test_target_transform_log1p():
    y = pd.Series([0.0, 10.0, 50.0, 100.0])
    y_log = transform_target(y, strategy="log1p")

    assert np.isclose(y_log.iloc[0], 0.0)
    assert np.isclose(y_log.iloc[1], np.log1p(10.0))


def test_target_inverse_transform_exact():
    y_raw = np.array([12.5, 45.0, 95.0, 250.0, 1500.0])
    y_log = transform_target(y_raw, strategy="log1p")
    y_recovered = inverse_transform_target(y_log, strategy="log1p")

    assert np.allclose(y_raw, y_recovered, atol=1e-5)


def test_target_inverse_transform_scalar():
    val = 75.0
    val_log = transform_target(val, strategy="log1p")
    val_rec = inverse_transform_target(val_log, strategy="log1p")

    assert np.isclose(val, val_rec, atol=1e-5)


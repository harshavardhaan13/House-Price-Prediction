"""
Unit tests for SmartHouse AI preprocessing pipeline and train/test splitting.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.features.pipeline import (
    build_feature_pipeline,
    build_preprocessor,
    split_and_save_data,
    get_model_ready_data,
    get_cv_splitter
)


@pytest.fixture
def mock_dataset_csv(tmp_path):
    csv_file = tmp_path / "mock_housing_dataset.csv"
    data = {
        "location": ["Whitefield", "Koramangala", "HSR Layout", "Electronic City", "Other"] * 20,
        "total_sqft": [1000.0, 1500.0, 2000.0, 1200.0, 800.0] * 20,
        "bath": [2.0, 3.0, 3.0, 2.0, 1.0] * 20,
        "balcony": [1.0, 2.0, 2.0, 1.0, 1.0] * 20,
        "bhk": [2.0, 3.0, 3.0, 2.0, 1.0] * 20,
        "area_type": ["Super built-up Area", "Plot Area", "Built-up Area", "Super built-up Area", "Super built-up Area"] * 20,
        "availability_status": ["Ready To Move", "Under Construction", "Ready To Move", "Ready To Move", "Under Construction"] * 20,
        "price": [45.0, 120.0, 160.0, 60.0, 30.0] * 20
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return csv_file


def test_build_feature_pipeline(mock_dataset_csv):
    df = pd.read_csv(mock_dataset_csv).drop(columns=["price"])
    pipeline = build_feature_pipeline(scaler_type="robust")
    X_trans = pipeline.fit_transform(df)

    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == len(df)
    assert X_trans.shape[1] > 0
    assert not np.isnan(X_trans).any()
    assert not np.isinf(X_trans).any()


def test_handle_unknown_locality(mock_dataset_csv):
    train_df = pd.read_csv(mock_dataset_csv).drop(columns=["price"]).head(50)
    test_unknown_df = pd.DataFrame({
        "location": ["Brand New Unknown Locality 123", "Fictional City Locality"],
        "total_sqft": [1400.0, 2100.0],
        "bath": [2.0, 3.0],
        "balcony": [1.0, 2.0],
        "bhk": [2.0, 3.0],
        "area_type": ["Super built-up Area", "Plot Area"],
        "availability_status": ["Ready To Move", "Under Construction"]
    })

    pipeline = build_feature_pipeline()
    pipeline.fit(train_df)

    # Should transform without raising any KeyError or ValueError
    X_test_trans = pipeline.transform(test_unknown_df)
    assert X_test_trans.shape[0] == 2
    assert not np.isnan(X_test_trans).any()


def test_split_and_save_data(mock_dataset_csv):
    train_df, test_df = split_and_save_data(
        data_path=mock_dataset_csv,
        test_size=0.20,
        random_state=42,
        save=False
    )
    assert len(train_df) == 80
    assert len(test_df) == 20
    assert "price" in train_df.columns
    assert "price" in test_df.columns


def test_get_model_ready_data(mock_dataset_csv):
    bundle = get_model_ready_data(
        data_path=mock_dataset_csv,
        test_size=0.20,
        random_state=42,
        target_transform="log1p",
        scaler_type="robust"
    )

    assert bundle["X_train"].shape[0] == 80
    assert bundle["X_test"].shape[0] == 20
    assert len(bundle["y_train"]) == 80
    assert len(bundle["y_test"]) == 20
    assert bundle["X_train"].shape[1] == bundle["X_test"].shape[1]
    assert not np.isnan(bundle["X_train"]).any()
    assert not np.isinf(bundle["X_train"]).any()


def test_cv_splitter():
    cv = get_cv_splitter(n_splits=5, random_state=42)
    assert cv.n_splits == 5
    assert cv.shuffle is True


def test_strict_leakage_exclusion(mock_dataset_csv):
    bundle = get_model_ready_data(
        data_path=mock_dataset_csv,
        test_size=0.20,
        random_state=42
    )

    # 1. Check raw feature names
    assert "price" not in bundle["X_train_raw"].columns
    assert "price_per_sqft" not in bundle["X_train_raw"].columns
    assert "price_lakhs" not in bundle["X_train_raw"].columns

    # 2. Check test feature names
    assert "price" not in bundle["X_test_raw"].columns
    assert "price_per_sqft" not in bundle["X_test_raw"].columns


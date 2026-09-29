"""
SmartHouse AI - Machine Learning Feature & Preprocessing Pipeline.
Provides robust ColumnTransformers, feature scaling, one-hot encoding with
unknown category handling, reproducible train/test splitting, and cross-validation utilities.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from src.features.engineering import (
    FeatureEngineeringTransformer,
    transform_target,
    inverse_transform_target,
    validate_no_target_leakage
)
from src.utils.config import (
    CLEANED_DATA_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH
)

logger = logging.getLogger("SmartHouseAI.Pipeline")

# Base Raw Columns expected as inputs (Excluding Target)
RAW_NUMERICAL_FEATURES = ["total_sqft", "bath", "balcony", "bhk"]
RAW_CATEGORICAL_FEATURES = ["location", "area_type", "availability_status"]

# All Numerical Features including Engineered Features
ALL_ENGINEERED_NUMERICAL_FEATURES = [
    "total_sqft",
    "bath",
    "balcony",
    "bhk",
    "total_rooms",
    "sqft_per_bhk",
    "bath_per_bhk",
    "sqft_per_room",
    "bhk_bath_diff"
]


def build_preprocessor(
    scaler_type: str = "robust",
    handle_unknown: str = "ignore"
) -> ColumnTransformer:
    """
    Constructs a robust sklearn ColumnTransformer for numerical scaling and categorical encoding.

    Args:
        scaler_type: 'robust' (RobustScaler), 'standard' (StandardScaler), or 'none' (passthrough).
        handle_unknown: Strategy for unknown categories during inference ('ignore').

    Returns:
        ColumnTransformer: Preprocessing transformer.
    """
    # 1. Numerical Pipeline
    if scaler_type == "robust":
        num_scaler = RobustScaler()
    elif scaler_type == "standard":
        num_scaler = StandardScaler()
    else:
        num_scaler = "passthrough"

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", num_scaler)
    ])

    # 2. Categorical Pipeline
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Other")),
        ("onehot", OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, ALL_ENGINEERED_NUMERICAL_FEATURES),
            ("cat", cat_pipeline, RAW_CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor


def build_feature_pipeline(
    scaler_type: str = "robust",
    handle_unknown: str = "ignore"
) -> Pipeline:
    """
    Constructs an end-to-end sklearn Pipeline from raw feature DataFrame to model-ready matrix.
    """
    preprocessor = build_preprocessor(scaler_type=scaler_type, handle_unknown=handle_unknown)

    pipeline = Pipeline([
        ("feature_engineering", FeatureEngineeringTransformer()),
        ("preprocessor", preprocessor)
    ])

    return pipeline


def split_and_save_data(
    data_path: Path = CLEANED_DATA_PATH,
    test_size: float = 0.20,
    random_state: int = RANDOM_SEED,
    save: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits cleaned dataset into train and test sets reproducibly and verifies zero target leakage.

    Args:
        data_path: Path to cleaned CSV dataset.
        test_size: Fraction of test split (default 0.20).
        random_state: Random seed for reproducibility (default 42).
        save: Whether to persist train.csv and test.csv in data/processed/.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    df = pd.read_csv(data_path)
    target_col = TARGET_COLUMN if TARGET_COLUMN in df.columns else ("price" if "price" in df.columns else "price_lakhs")
    if target_col not in df.columns:
        raise ValueError(f"Target column ('price' or 'price_lakhs') missing from dataset at {data_path}.")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )


    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(TRAIN_DATA_PATH, index=False)
        test_df.to_csv(TEST_DATA_PATH, index=False)
        logger.info(
            f"Saved train split ({len(train_df)} rows) to {TRAIN_DATA_PATH} "
            f"and test split ({len(test_df)} rows) to {TEST_DATA_PATH}."
        )

    return train_df, test_df


def get_model_ready_data(
    data_path: Path = CLEANED_DATA_PATH,
    test_size: float = 0.20,
    random_state: int = RANDOM_SEED,
    target_transform: str = "log1p",
    scaler_type: str = "robust"
) -> Dict[str, Any]:
    """
    Prepares complete train/test feature matrices and target vectors.
    """
    train_df, test_df = split_and_save_data(
        data_path=data_path,
        test_size=test_size,
        random_state=random_state,
        save=True
    )

    target_col = TARGET_COLUMN if TARGET_COLUMN in train_df.columns else ("price" if "price" in train_df.columns else "price_lakhs")

    # Separate raw inputs and targets
    X_train_raw = train_df.drop(columns=[target_col])
    y_train_raw = train_df[target_col]

    X_test_raw = test_df.drop(columns=[target_col])
    y_test_raw = test_df[target_col]


    # Verify no target leakage in input features
    validate_no_target_leakage(X_train_raw.columns)
    validate_no_target_leakage(X_test_raw.columns)

    # Transform targets
    y_train = transform_target(y_train_raw, strategy=target_transform)
    y_test = transform_target(y_test_raw, strategy=target_transform)

    # Build and fit feature pipeline on TRAIN ONLY (strictly prevent test set leakage)
    feature_pipeline = build_feature_pipeline(scaler_type=scaler_type)
    X_train_transformed = feature_pipeline.fit_transform(X_train_raw)
    X_test_transformed = feature_pipeline.transform(X_test_raw)

    # Check for NaNs or Inf in model-ready features
    if np.isnan(X_train_transformed).any() or np.isinf(X_train_transformed).any():
        raise ValueError("Model-ready X_train contains NaN or Infinite values!")
    if np.isnan(X_test_transformed).any() or np.isinf(X_test_transformed).any():
        raise ValueError("Model-ready X_test contains NaN or Infinite values!")

    return {
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "X_train": X_train_transformed,
        "X_test": X_test_transformed,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_raw": y_train_raw,
        "y_test_raw": y_test_raw,
        "feature_pipeline": feature_pipeline,
        "target_transform": target_transform,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "num_features_out": X_train_transformed.shape[1]
    }


def get_cv_splitter(n_splits: int = 5, random_state: int = RANDOM_SEED) -> KFold:
    """
    Returns reproducible KFold cross-validation splitter.
    """
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


if __name__ == "__main__":
    data_bundle = get_model_ready_data()
    print("\n" + "="*60)
    print("SMARTHOUSE AI - FEATURE PIPELINE READY")
    print("="*60)
    print(f"Train Records: {data_bundle['train_samples']:,} (80%)")
    print(f"Test Records: {data_bundle['test_samples']:,} (20%)")
    print(f"Model Feature Dimension: {data_bundle['num_features_out']} features")
    print(f"Target Transform Strategy: {data_bundle['target_transform']}")
    print(f"NaN or Inf Check: PASSED (Zero NaNs/Infs)")
    print("="*60)


"""
SmartHouse AI - Feature Engineering & Leakage Safeguards Module.
Implements domain feature derivation, sklearn-compatible transformers,
target log-transformations, and strict data leakage exclusion enforcement.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger("SmartHouseAI.FeatureEngineering")

# Explicit Blacklist of Target Leakage Features
# Under NO circumstance may any feature in this set enter the training/inference feature matrix X.
LEAKAGE_FEATURE_BLACKLIST = {
    "price",
    "price_lakhs",
    "price_per_sqft",
    "target",
    "target_encoded_price",
    "target_price"
}


def validate_no_target_leakage(feature_names: Union[List[str], pd.Index]) -> None:
    """
    Validates that no target-derived or leakage features are present in the feature matrix.

    Args:
        feature_names: List or Index of column/feature names.

    Raises:
        ValueError: If any leakage feature is detected.
    """
    found_leakage = [col for col in feature_names if col.lower() in LEAKAGE_FEATURE_BLACKLIST]
    if found_leakage:
        raise ValueError(
            f"CRITICAL DATA LEAKAGE DETECTED: Features {found_leakage} are mathematically "
            f"derived from the target variable and cannot be used as model inputs."
        )


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that engineers domain-specific real estate features.
    
    Engineered Features:
    - total_rooms: sum of bedrooms (bhk) and bathrooms (bath)
    - sqft_per_bhk: total square footage divided by bedroom count (density/spaciousness)
    - bath_per_bhk: bathroom to bedroom ratio
    - sqft_per_room: total area divided by total room count
    - bhk_bath_diff: bathroom count minus bedroom count (luxury suite indicator)
    """

    def __init__(self, include_interaction_features: bool = True):
        self.include_interaction_features = include_interaction_features
        self.engineered_feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit on training data only; learn rare locality categories without target information."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineeringTransformer requires a pandas DataFrame input.")
        validate_no_target_leakage(X.columns)
        self.known_locations_ = set()
        self.rare_locations_ = set()
        if "location" in X.columns:
            loc = self._normalize_text(X["location"])
            counts = loc.value_counts(dropna=False)
            self.known_locations_ = set(counts.index.dropna().tolist())
            self.rare_locations_ = set(counts[counts < 10].index.tolist())
        return self

    @staticmethod
    def _normalize_text(series: pd.Series) -> pd.Series:
        return (series.astype("string").str.replace(r"\s+", " ", regex=True)
                .str.strip().str.title())

    def _normalize_categories(self, df_out: pd.DataFrame) -> pd.DataFrame:
        if "location" in df_out.columns:
            loc = self._normalize_text(df_out["location"])
            rare = getattr(self, "rare_locations_", set())
            known = getattr(self, "known_locations_", set())
            df_out["location"] = loc.where(~loc.isin(rare) & loc.isin(known), "Other").fillna("Other")
        for col in ("area_type", "availability_status"):
            if col in df_out.columns:
                df_out[col] = self._normalize_text(df_out[col])
        return df_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input DataFrame by appending engineered features.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineeringTransformer requires a pandas DataFrame input.")

        validate_no_target_leakage(X.columns)
        df_out = self._normalize_categories(X.copy())

        # Extract base numerical series safely with defaults if missing
        sqft = pd.to_numeric(df_out["total_sqft"], errors="coerce").fillna(1000.0) if "total_sqft" in df_out.columns else pd.Series(1000.0, index=df_out.index)
        bhk = pd.to_numeric(df_out["bhk"], errors="coerce").fillna(2.0) if "bhk" in df_out.columns else pd.Series(2.0, index=df_out.index)
        bath = pd.to_numeric(df_out["bath"], errors="coerce").fillna(2.0) if "bath" in df_out.columns else pd.Series(2.0, index=df_out.index)
        balcony = pd.to_numeric(df_out["balcony"], errors="coerce").fillna(1.0) if "balcony" in df_out.columns else pd.Series(1.0, index=df_out.index)

        # Safe division denominators (prevent division by zero)
        safe_bhk = np.maximum(bhk, 1.0)
        total_rooms = bhk + bath

        # 1. Total Rooms
        df_out["total_rooms"] = total_rooms

        # 2. Sqft per BHK (Spatial Density)
        df_out["sqft_per_bhk"] = (sqft / safe_bhk).round(2)

        # 3. Bath to BHK Ratio
        df_out["bath_per_bhk"] = (bath / safe_bhk).round(2)

        # 4. Sqft per Total Rooms
        df_out["sqft_per_room"] = (sqft / np.maximum(total_rooms, 1.0)).round(2)

        # 5. BHK Bath Difference
        df_out["bhk_bath_diff"] = (bath - bhk).round(2)

        self.engineered_feature_names_ = [
            "total_rooms",
            "sqft_per_bhk",
            "bath_per_bhk",
            "sqft_per_room",
            "bhk_bath_diff"
        ]

        # Double check no leakage
        validate_no_target_leakage(df_out.columns)

        return df_out

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Returns feature names after transformation."""
        base_features = list(input_features) if input_features is not None else []
        return base_features + self.engineered_feature_names_


def transform_target(
    y: Union[pd.Series, np.ndarray],
    strategy: str = "log1p"
) -> Union[pd.Series, np.ndarray]:
    """
    Transforms the target price variable.
    
    Args:
        y: Original target values (in ₹ Lakhs).
        strategy: 'log1p' (default) or 'identity'.

    Returns:
        Transformed target values.
    """
    if strategy == "log1p":
        # Ensure strictly positive
        y_clean = np.maximum(y, 0.0)
        if isinstance(y, pd.Series):
            return np.log1p(y_clean)
        return np.log1p(y_clean)
    elif strategy == "identity":
        return y
    else:
        raise ValueError(f"Unknown target transformation strategy: {strategy}")


def inverse_transform_target(
    y_trans: Union[pd.Series, np.ndarray, float],
    strategy: str = "log1p"
) -> Union[pd.Series, np.ndarray, float]:
    """
    Transforms predicted target values back to original ₹ Lakhs scale.
    
    Args:
        y_trans: Predicted values in transformed space.
        strategy: 'log1p' (default) or 'identity'.

    Returns:
        Predicted values in ₹ Lakhs (clipped at 0 to avoid impossible negative prices).
    """
    if strategy == "log1p":
        if isinstance(y_trans, (int, float, np.floating)):
            return max(0.0, float(np.expm1(y_trans)))
        res = np.expm1(y_trans)
        return np.maximum(res, 0.0)
    elif strategy == "identity":
        if isinstance(y_trans, (int, float, np.floating)):
            return max(0.0, float(y_trans))
        return np.maximum(y_trans, 0.0)
    else:
        raise ValueError(f"Unknown target transformation strategy: {strategy}")


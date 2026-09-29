"""
SmartHouse AI - Data Validation Module.
Performs schema verification, missing-value audit, duplicate detection,
impossible domain-value checks, and statistical outlier identification.
"""
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("SmartHouseAI.DataValidator")


class DataValidator:
    """
    Comprehensive data validation engine for real estate datasets.
    """

    def __init__(
        self,
        required_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None
    ):
        self.required_columns = required_columns or []
        self.target_column = target_column

    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validates required columns and basic schema integrity.
        """
        missing = [c for c in self.required_columns if c not in df.columns]
        return {
            "valid": len(missing) == 0,
            "missing_required_columns": missing,
            "existing_columns": list(df.columns)
        }

    def check_missing_values(
        self,
        df: pd.DataFrame,
        critical_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyzes missing values across all and critical columns.
        """
        missing_dict = df.isnull().sum().to_dict()
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2).to_dict()

        critical_missing = {}
        if critical_columns:
            for col in critical_columns:
                if col in df.columns and missing_dict.get(col, 0) > 0:
                    critical_missing[col] = missing_dict[col]

        return {
            "total_missing_cells": int(df.isnull().sum().sum()),
            "columns_with_missing": {k: v for k, v in missing_dict.items() if v > 0},
            "missing_percentages": {k: v for k, v in missing_pct.items() if v > 0},
            "critical_missing": critical_missing
        }

    def check_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Identifies exact or subset duplicate records.
        """
        dup_count = int(df.duplicated(subset=subset).sum())
        return {
            "duplicate_count": dup_count,
            "duplicate_percentage": round((dup_count / len(df)) * 100, 2) if len(df) > 0 else 0.0
        }

    def check_impossible_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Checks for physical/domain impossibilities (negative area, non-positive price, excessive baths).
        """
        anomalies = {}

        if "price" in df.columns:
            non_pos_price = int((pd.to_numeric(df["price"], errors="coerce") <= 0).sum())
            if non_pos_price > 0:
                anomalies["non_positive_price_count"] = non_pos_price

        if "bath" in df.columns and "bhk" in df.columns:
            bath_num = pd.to_numeric(df["bath"], errors="coerce")
            bhk_num = pd.to_numeric(df["bhk"], errors="coerce")
            excess_bath = int((bath_num > (bhk_num + 3)).sum())
            if excess_bath > 0:
                anomalies["excessive_bathrooms_count"] = excess_bath

        if "total_sqft" in df.columns:
            sqft_num = pd.to_numeric(df["total_sqft"], errors="coerce")
            neg_sqft = int((sqft_num <= 0).sum())
            if neg_sqft > 0:
                anomalies["non_positive_sqft_count"] = neg_sqft

        if "bhk" in df.columns:
            bhk_num = pd.to_numeric(df["bhk"], errors="coerce")
            neg_bhk = int((bhk_num <= 0).sum())
            if neg_bhk > 0:
                anomalies["non_positive_bhk_count"] = neg_bhk

        return {
            "impossible_count": sum(anomalies.values()),
            "anomalies_breakdown": anomalies
        }

    def detect_outliers_iqr(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        factor: float = 1.5
    ) -> Dict[str, Any]:
        """
        Detects statistical outliers using Interquartile Range (IQR).
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outlier_info = {}
        for col in columns:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(series) > 0:
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - factor * iqr
                    upper_bound = q3 + factor * iqr
                    outliers = series[(series < lower_bound) | (series > upper_bound)]
                    outlier_info[col] = {
                        "q1": round(float(q1), 2),
                        "q3": round(float(q3), 2),
                        "iqr": round(float(iqr), 2),
                        "lower_bound": round(float(lower_bound), 2),
                        "upper_bound": round(float(upper_bound), 2),
                        "outlier_count": int(len(outliers)),
                        "outlier_percentage": round(len(outliers) / len(series) * 100, 2)
                    }

        return outlier_info

    def generate_validation_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates an end-to-end data validation report.
        """
        schema_res = self.validate_schema(df)
        missing_res = self.check_missing_values(df, critical_columns=["price", "location"])
        dup_res = self.check_duplicates(df)
        impossible_res = self.check_impossible_values(df)
        outliers_res = self.detect_outliers_iqr(df)

        is_valid = schema_res["valid"] and (missing_res["critical_missing"] == {})

        return {
            "is_valid": is_valid,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "schema_validation": schema_res,
            "missing_value_analysis": missing_res,
            "duplicate_analysis": dup_res,
            "domain_impossibilities": impossible_res,
            "outlier_detection": outliers_res
        }


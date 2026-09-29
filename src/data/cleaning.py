"""
SmartHouse AI - Data Cleaning Pipeline Module.
Implements robust parsing, imputation, categorical normalization,
and domain-specific anomaly filtering while preserving raw data.
"""
import logging
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger("SmartHouseAI.DataCleaner")


class DataCleaner:
    """
    Robust cleaning and transformation pipeline for real estate records.
    """

    @staticmethod
    def parse_bhk(val: Any) -> Optional[int]:
        """
        Extracts integer BHK / bedroom count from string representations (e.g. '2 BHK', '4 Bedroom').
        """
        if pd.isna(val):
            return None
        val_str = str(val).strip()
        match = re.search(r"(\d+)", val_str)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def parse_sqft(val: Any) -> Optional[float]:
        """
        Standardizes total_sqft from strings, ranges, and area units into square feet float.
        """
        if pd.isna(val):
            return None
        val_str = str(val).strip()

        # Case 1: Direct float conversion
        try:
            return float(val_str)
        except ValueError:
            pass

        # Case 2: Range like '1133 - 1384'
        if '-' in val_str:
            parts = val_str.split('-')
            try:
                if len(parts) == 2:
                    low = float(parts[0].strip())
                    high = float(parts[1].strip())
                    return (low + high) / 2.0
            except ValueError:
                pass

        # Case 3: Unit conversions
        val_lower = val_str.lower()
        if 'sq. meter' in val_lower or 'sq.meter' in val_lower:
            num = re.findall(r"[\d\.]+", val_str)
            if num:
                return float(num[0]) * 10.7639
        elif 'sq. yard' in val_lower or 'sq.yard' in val_lower:
            num = re.findall(r"[\d\.]+", val_str)
            if num:
                return float(num[0]) * 9.0
        elif 'acre' in val_lower:
            num = re.findall(r"[\d\.]+", val_str)
            if num:
                return float(num[0]) * 43560.0
        elif 'guntha' in val_lower:
            num = re.findall(r"[\d\.]+", val_str)
            if num:
                return float(num[0]) * 1089.0
        elif 'perch' in val_lower:
            num = re.findall(r"[\d\.]+", val_str)
            if num:
                return float(num[0]) * 272.25

        # Case 4: Any other embedded float
        num_match = re.findall(r"[\d\.]+", val_str)
        if num_match:
            try:
                return float(num_match[0])
            except ValueError:
                return None

        return None

    @staticmethod
    def clean_availability(val: Any) -> str:
        """
        Standardizes availability into 'Ready To Move' vs 'Under Construction'.
        """
        if pd.isna(val):
            return "Ready To Move"
        val_str = str(val).strip()
        if val_str.lower() == "ready to move":
            return "Ready To Move"
        return "Under Construction"

    @staticmethod
    def clean_locations(series: pd.Series, min_count: int = 10) -> pd.Series:
        """
        Normalizes location strings (strips whitespace, title case) and aggregates rare locations.
        """
        cleaned = series.fillna("Other").astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        cleaned = cleaned.str.title()

        location_counts = cleaned.value_counts()
        frequent_locations = set(location_counts[location_counts >= min_count].index)

        return cleaned.apply(lambda x: x if x in frequent_locations else "Other")

    def clean_dataset(
        self,
        df: pd.DataFrame,
        min_location_frequency: int = 10,
        remove_duplicates: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Runs the full cleaning pipeline on the dataframe without modifying original.
        """
        initial_rows = len(df)
        clean_df = df.copy()

        # 1. Remove exact duplicate rows
        duplicates_removed = 0
        if remove_duplicates:
            dups = clean_df.duplicated().sum()
            clean_df = clean_df.drop_duplicates().reset_index(drop=True)
            duplicates_removed = int(dups)

        # 2. Parse BHK
        if "size" in clean_df.columns:
            clean_df["bhk"] = clean_df["size"].apply(self.parse_bhk)
        elif "bhk" not in clean_df.columns:
            clean_df["bhk"] = 2

        # 3. Parse total_sqft
        if "total_sqft" in clean_df.columns:
            clean_df["total_sqft"] = clean_df["total_sqft"].apply(self.parse_sqft)

        # 4. Standardize Availability
        if "availability" in clean_df.columns:
            clean_df["availability_status"] = clean_df["availability"].apply(self.clean_availability)
        else:
            clean_df["availability_status"] = "Ready To Move"

        # 5. Clean Location
        if "location" in clean_df.columns:
            clean_df["location"] = self.clean_locations(clean_df["location"], min_count=min_location_frequency)
        else:
            clean_df["location"] = "Other"

        # 6. Clean Area Type
        if "area_type" in clean_df.columns:
            clean_df["area_type"] = clean_df["area_type"].fillna("Super built-up Area").astype(str).str.strip()
        else:
            clean_df["area_type"] = "Super built-up Area"

        # 7. Drop rows with null essential values (bhk, total_sqft, price)
        null_essentials_before = len(clean_df)
        clean_df = clean_df.dropna(subset=["bhk", "total_sqft", "price"]).reset_index(drop=True)
        dropped_null_essentials = null_essentials_before - len(clean_df)

        # 8. Impute Bathrooms based on BHK median
        clean_df["bath"] = clean_df["bath"].fillna(clean_df.groupby("bhk")["bath"].transform("median"))
        clean_df["bath"] = clean_df["bath"].fillna(clean_df["bhk"]).astype(int)

        # 9. Impute Balconies based on BHK median (or 1)
        clean_df["balcony"] = clean_df["balcony"].fillna(clean_df.groupby("bhk")["balcony"].transform("median"))
        clean_df["balcony"] = clean_df["balcony"].fillna(1).astype(int)

        # 10. Filter Domain Outliers / Errors (sqft/bhk >= 200, bath <= bhk + 2, price > 0, total_sqft <= 25000)
        anomalies_mask = (
            (clean_df["total_sqft"] / clean_df["bhk"] >= 200) &
            (clean_df["total_sqft"] <= 25000) &
            (clean_df["bath"] <= clean_df["bhk"] + 2) &
            (clean_df["price"] > 0) &
            (clean_df["bhk"] <= 12)
        )

        filtered_anomalies_count = int((~anomalies_mask).sum())
        clean_df = clean_df[anomalies_mask].reset_index(drop=True)

        final_cols = [
            "location",
            "total_sqft",
            "bath",
            "balcony",
            "bhk",
            "area_type",
            "availability_status",
            "price"
        ]
        available_cols = [c for c in final_cols if c in clean_df.columns]
        clean_df = clean_df[available_cols]

        report = {
            "initial_rows": initial_rows,
            "duplicates_removed": duplicates_removed,
            "dropped_null_essentials": dropped_null_essentials,
            "domain_anomalies_filtered": filtered_anomalies_count,
            "final_cleaned_rows": len(clean_df),
            "final_columns": list(clean_df.columns),
            "data_retention_rate_pct": round((len(clean_df) / initial_rows) * 100, 2)
        }
        logger.info(f"Cleaning finished: {report}")
        return clean_df, report


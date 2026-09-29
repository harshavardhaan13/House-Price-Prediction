"""
SmartHouse AI - Data Quality & Diagnostics Reporter.
Generates comprehensive quality audits, distributional statistics,
missing/duplicate analyses, and outputs structured JSON/console summaries.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd

from src.data.loader import load_csv_data, inspect_dataframe
from src.data.validation import DataValidator
from src.data.cleaning import DataCleaner
from src.utils.config import RAW_DATA_PATH, CLEANED_DATA_PATH, DOCS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SmartHouseAI.DataReport")


def generate_data_quality_report(
    raw_path: Path = RAW_DATA_PATH,
    save_cleaned: bool = True
) -> Dict[str, Any]:
    """
    Executes full loading, validation, cleaning, and quality diagnostics reporting.
    """
    logger.info(f"Generating Data Quality Report for raw dataset: {raw_path}")

    # 1. Load Raw
    raw_df = load_csv_data(raw_path)
    raw_inspection = inspect_dataframe(raw_df)

    # 2. Validate Raw
    validator = DataValidator(
        required_columns=["location", "total_sqft", "price"],
        target_column="price"
    )
    raw_validation = validator.generate_validation_report(raw_df)

    # 3. Clean
    cleaner = DataCleaner()
    cleaned_df, cleaning_summary = cleaner.clean_dataset(raw_df)

    if save_cleaned:
        CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)
        logger.info(f"Saved processed cleaned dataset to: {CLEANED_DATA_PATH}")

    # 4. Cleaned Diagnostics
    cleaned_inspection = inspect_dataframe(cleaned_df)
    cleaned_validation = validator.generate_validation_report(cleaned_df)

    # 5. Target Variable Statistics (EDA insights)
    target_stats = {
        "mean_price_lakhs": round(float(cleaned_df["price"].mean()), 2),
        "median_price_lakhs": round(float(cleaned_df["price"].median()), 2),
        "std_price_lakhs": round(float(cleaned_df["price"].std()), 2),
        "min_price_lakhs": round(float(cleaned_df["price"].min()), 2),
        "max_price_lakhs": round(float(cleaned_df["price"].max()), 2),
        "skewness": round(float(cleaned_df["price"].skew()), 2)
    }

    # 6. Locality Overview
    top_locations = cleaned_df["location"].value_counts().head(10).to_dict()

    report = {
        "dataset_source": "Bengaluru House Price Dataset (Kaggle / Public Data)",
        "raw_dataset_summary": raw_inspection,
        "raw_validation_summary": raw_validation,
        "cleaning_operations": cleaning_summary,
        "cleaned_dataset_summary": cleaned_inspection,
        "cleaned_validation_summary": cleaned_validation,
        "target_statistics": target_stats,
        "top_locations": top_locations
    }

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report_json_path = DOCS_DIR / "data_quality_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Quality report exported to: {report_json_path}")
    return report


if __name__ == "__main__":
    rep = generate_data_quality_report()
    print("\n" + "="*60)
    print("SMARTHOUSE AI - DATA QUALITY & ENGINEERING SUMMARY")
    print("="*60)
    print(f"Raw Records: {rep['raw_dataset_summary']['row_count']}")
    print(f"Duplicates Removed: {rep['cleaning_operations']['duplicates_removed']}")
    print(f"Domain Outliers Filtered: {rep['cleaning_operations']['domain_anomalies_filtered']}")
    print(f"Cleaned Records Generated: {rep['cleaning_operations']['final_cleaned_rows']} (Retention: {rep['cleaning_operations']['data_retention_rate_pct']}%)")
    print(f"Target Price Median: INR {rep['target_statistics']['median_price_lakhs']} Lakhs")
    print(f"Target Price Range: INR {rep['target_statistics']['min_price_lakhs']} Lakhs to INR {rep['target_statistics']['max_price_lakhs']} Lakhs")
    print(f"Cleaned Schema Valid: {rep['cleaned_validation_summary']['is_valid']}")
    print("="*60)



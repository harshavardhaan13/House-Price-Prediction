"""
SmartHouse AI - Data Loading Module.
Reusable functions for loading, inspecting, and basic validation of tabular real estate data.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

logger = logging.getLogger("SmartHouseAI.DataLoader")


def load_csv_data(
    filepath: str | Path,
    required_columns: Optional[List[str]] = None,
    encoding: str = "utf-8"
) -> pd.DataFrame:
    """
    Loads CSV data with path validation and schema checking.

    Args:
        filepath: Path to CSV file.
        required_columns: Optional list of mandatory columns.
        encoding: File encoding (default 'utf-8').

    Returns:
        pd.DataFrame: Loaded dataframe.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is empty or missing required columns.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at path: {path}")

    try:
        df = pd.read_csv(path, encoding=encoding)
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {path}")
    except Exception as e:
        raise ValueError(f"Failed to read CSV at {path}: {str(e)}")

    if df.empty:
        raise ValueError(f"Loaded DataFrame contains 0 rows: {path}")

    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset missing required columns: {missing_cols}")

    logger.info(f"Loaded dataset from {path.name} with shape {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def inspect_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes structural summary and diagnostics of a DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    null_counts = df.isnull().sum().to_dict()
    null_percentages = (df.isnull().sum() / len(df) * 100).round(2).to_dict() if len(df) > 0 else {}

    return {
        "shape": df.shape,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": null_counts,
        "missing_percentages": null_percentages,
        "duplicate_count": int(df.duplicated().sum()),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    }
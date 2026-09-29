"""
Unit tests for SmartHouse AI data loader module.
"""
import pytest
import pandas as pd
from pathlib import Path
from src.data.loader import load_csv_data, inspect_dataframe


@pytest.fixture
def sample_csv(tmp_path):
    csv_file = tmp_path / "sample_housing.csv"
    data = {
        "location": ["Whitefield", "Koramangala", "Indiranagar"],
        "total_sqft": ["1200", "1500", "1800"],
        "bath": [2.0, 3.0, 3.0],
        "balcony": [1.0, 2.0, 2.0],
        "price": [75.0, 120.0, 160.0]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def empty_csv(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")
    return empty_file


def test_load_csv_success(sample_csv):
    df = load_csv_data(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "price" in df.columns


def test_load_csv_missing_file():
    with pytest.raises(FileNotFoundError):
        load_csv_data("non_existent_path.csv")


def test_load_csv_empty_file(empty_csv):
    with pytest.raises(ValueError):
        load_csv_data(empty_csv)


def test_load_csv_required_columns(sample_csv):
    df = load_csv_data(sample_csv, required_columns=["location", "price"])
    assert len(df) == 3

    with pytest.raises(ValueError, match="missing required columns"):
        load_csv_data(sample_csv, required_columns=["location", "non_existent_column"])


def test_inspect_dataframe(sample_csv):
    df = load_csv_data(sample_csv)
    report = inspect_dataframe(df)

    assert report["row_count"] == 3
    assert report["column_count"] == 5
    assert report["duplicate_count"] == 0
    assert "price" in report["missing_counts"]
    assert report["memory_usage_mb"] >= 0.0


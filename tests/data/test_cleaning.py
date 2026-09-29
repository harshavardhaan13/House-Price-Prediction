"""
Unit tests for SmartHouse AI data cleaning module.
"""
import pytest
import pandas as pd
import numpy as np
from src.data.cleaning import DataCleaner


@pytest.fixture
def cleaner():
    return DataCleaner()


@pytest.fixture
def raw_test_df():
    return pd.DataFrame({
        "area_type": ["Super built-up  Area", "Plot  Area", "Built-up  Area", "Super built-up  Area", "Super built-up  Area"],
        "availability": ["Ready To Move", "19-Dec", "Ready To Move", "Ready To Move", "Ready To Move"],
        "location": ["Whitefield ", " Electronic City ", "Whitefield", "Unknown Small Locality", "Whitefield "],
        "size": ["2 BHK", "4 Bedroom", "3 BHK", "2 BHK", "2 BHK"],
        "society": ["Pha 3cs", None, "GrrvaGr", None, "Pha 3cs"],
        "total_sqft": ["1056", "2600", "1133 - 1384", "1000", "1056"],
        "bath": [2.0, 5.0, 3.0, 2.0, 2.0],
        "balcony": [1.0, 3.0, np.nan, 1.0, 1.0],
        "price": [39.07, 120.0, 62.0, 45.0, 39.07]  # row 4 is duplicate of row 0
    })


def test_parse_bhk(cleaner):
    assert cleaner.parse_bhk("2 BHK") == 2
    assert cleaner.parse_bhk("4 Bedroom") == 4
    assert cleaner.parse_bhk("1 RK") == 1
    assert cleaner.parse_bhk("10 BHK") == 10
    assert cleaner.parse_bhk(None) is None


def test_parse_sqft_standard(cleaner):
    assert cleaner.parse_sqft("1200") == 1200.0
    assert cleaner.parse_sqft("1500.5") == 1500.5


def test_parse_sqft_range(cleaner):
    assert cleaner.parse_sqft("1000 - 1200") == 1100.0
    assert cleaner.parse_sqft("2100 - 2850") == 2475.0


def test_parse_sqft_units(cleaner):
    # 1 sq.meter = ~10.7639 sq.ft
    assert abs(cleaner.parse_sqft("100Sq. Meter") - 1076.39) < 1.0
    # 1 sq.yard = 9 sq.ft
    assert cleaner.parse_sqft("100Sq. Yards") == 900.0


def test_clean_availability(cleaner):
    assert cleaner.clean_availability("Ready To Move") == "Ready To Move"
    assert cleaner.clean_availability("19-Dec") == "Under Construction"
    assert cleaner.clean_availability(None) == "Ready To Move"


def test_clean_dataset_end_to_end(cleaner, raw_test_df):
    clean_df, report = cleaner.clean_dataset(raw_test_df, min_location_frequency=2, remove_duplicates=True)

    assert isinstance(clean_df, pd.DataFrame)
    assert "price" in clean_df.columns
    assert "bhk" in clean_df.columns
    assert "total_sqft" in clean_df.columns
    assert clean_df["total_sqft"].dtype == float

    # Duplicate row was removed
    assert report["duplicates_removed"] >= 1

    # Rare location was mapped to 'Other'
    assert "Other" in clean_df["location"].values

    # Imputation filled nan balcony
    assert clean_df["balcony"].isnull().sum() == 0


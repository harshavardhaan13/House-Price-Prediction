"""
Central Configuration Module for SmartHouse AI.
Defines project paths, column schemas, hyperparameter search spaces,
and Indian metropolitan locality geospatial and valuation metadata.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_PATH = DATA_DIR / 'raw' / 'Bengaluru_House_Data.csv'
RAW_BACKUP_PATH = DATA_DIR / 'raw' / 'housing_data.csv'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / 'cleaned_housing_data.csv'
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / 'train.csv'
TEST_DATA_PATH = PROCESSED_DATA_DIR / 'test.csv'

MODELS_DIR = BASE_DIR / 'models'
BEST_MODEL_PATH = MODELS_DIR / 'best_model.joblib'
PREPROCESSOR_PATH = MODELS_DIR / 'preprocessor.joblib'
MODEL_METADATA_PATH = MODELS_DIR / 'metadata.json'
MODEL_COMPARISON_PATH = MODELS_DIR / 'model_comparison.json'

DATABASE_PATH = BASE_DIR / 'data' / 'smarthouse.db'
DOCS_DIR = BASE_DIR / 'docs'
VISUALIZATIONS_DIR = BASE_DIR / 'visualizations'


# Reproducibility
RANDOM_SEED = 42

# Target Variable
TARGET_COLUMN = 'price'

# Raw Features actually present in the Bengaluru dataset
NUMERICAL_FEATURES = [
    "total_sqft", "bath", "balcony", "bhk"
]
CATEGORICAL_FEATURES = [
    "location", "area_type", "availability_status"
]
ALL_RAW_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
ENGINEERED_NUMERICAL_FEATURES = [
    "total_rooms", "sqft_per_bhk", "bath_per_bhk",
    "sqft_per_room", "bhk_bath_diff"
]
ENGINEERED_CATEGORICAL_FEATURES = []

# Locality Metadata for Geospatial Intelligence & Pricing Benchmarks
LOCATION_METADATA = {
    'Koramangala': {'lat': 12.9352, 'lon': 77.6245, 'tier': 'Tier-1 Prime', 'base_sqft_price': 12500},
    'Indiranagar': {'lat': 12.9784, 'lon': 77.6408, 'tier': 'Tier-1 Prime', 'base_sqft_price': 13000},
    'Whitefield': {'lat': 12.9698, 'lon': 77.7500, 'tier': 'Tech Hub', 'base_sqft_price': 8500},
    'HSR Layout': {'lat': 12.9121, 'lon': 77.6446, 'tier': 'Tier-1 Prime', 'base_sqft_price': 11000},
    'Electronic City': {'lat': 12.8399, 'lon': 77.6770, 'tier': 'Tech Hub', 'base_sqft_price': 6200},
    'Bellandur': {'lat': 12.9304, 'lon': 77.6784, 'tier': 'Tech Hub', 'base_sqft_price': 9200},
    'Marathahalli': {'lat': 12.9591, 'lon': 77.6974, 'tier': 'Growth Corridor', 'base_sqft_price': 7800},
    'Hebbal': {'lat': 13.0358, 'lon': 77.5970, 'tier': 'North Hub', 'base_sqft_price': 9800},
    'Jayanagar': {'lat': 12.9308, 'lon': 77.5838, 'tier': 'Established Heritage', 'base_sqft_price': 14000},
    'JP Nagar': {'lat': 12.9063, 'lon': 77.5857, 'tier': 'Established Residential', 'base_sqft_price': 8900},
    'Sarjapur Road': {'lat': 12.9166, 'lon': 77.6833, 'tier': 'Growth Corridor', 'base_sqft_price': 7600},
    'Yelahanka': {'lat': 13.1007, 'lon': 77.5963, 'tier': 'North Hub', 'base_sqft_price': 6800},
    'Bannerghatta Road': {'lat': 12.8943, 'lon': 77.5983, 'tier': 'South Growth Corridor', 'base_sqft_price': 7200},
    'Malleshwaram': {'lat': 13.0031, 'lon': 77.5643, 'tier': 'Established Heritage', 'base_sqft_price': 13500},
    'Thanisandra': {'lat': 13.0547, 'lon': 77.6327, 'tier': 'North Hub', 'base_sqft_price': 7100}
}

PROPERTY_TYTES = ['Apartment', 'Independent House', 'Villa', 'Penthouse']
FUNNISHING_STATUSES = ['Unfurnished', 'Semi-Furnished', 'Fully-Furnished']

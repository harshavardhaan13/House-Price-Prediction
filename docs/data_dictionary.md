# SmartHouse AI — Data Dictionary

This document details all features within the **SmartHouse AI** real estate dataset, specifying their origin, mathematical units, definitions, domain transformations, and role in model prediction vs. exploratory data analysis.

---

## 1. Dataset Overview
- **Dataset Name**: Bengaluru Real Estate Housing Transactions
- **Source**: Public Kaggle / Open Benchmark Dataset (`Bengaluru_House_Data.csv`)
- **Raw Dimensions**: 13,320 rows × 9 columns
- **Cleaned Dimensions**: 12,019 rows × 8 columns
- **Target Variable**: `price` (in ₹ Lakhs INR)

---

## 2. Raw Features (Original Ingestion Schema)

| Feature Name | Raw Data Type | Unit | Description | Example Values | Used for Prediction? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `area_type` | `object` (string) | N/A | Real estate classification of built area structure. | `Super built-up Area`, `Plot Area`, `Built-up Area`, `Carpet Area` | **Yes** (Categorical) |
| `availability` | `object` (string) | Date / Status | Immediate availability or construction delivery timeline. | `Ready To Move`, `19-Dec`, `18-May` | **Transformed** into `availability_status` |
| `location` | `object` (string) | N/A | Locality / neighborhood name within Bengaluru metropolitan region. | `Whitefield`, `Koramangala`, `Electronic City`, `Sarjapur Road` | **Yes** (Categorical, normalized) |
| `size` | `object` (string) | Text / Count | Text description of room composition. | `2 BHK`, `4 Bedroom`, `3 BHK`, `1 RK` | **Transformed** into `bhk` (integer) |
| `society` | `object` (string) | N/A | Housing society or apartment complex name (~41% missing in raw data). | `Pha 3cs`, `GrrvaGr`, `Unknown` | **No** (High cardinality & missing rate) |
| `total_sqft` | `object` (mixed) | Sq. Feet | Total property area. In raw data, contains floats, ranges (`1133 - 1384`), and non-standard units (`Sq. Meter`, `Acres`, `Guntha`). | `1056`, `1133 - 1384`, `34.46Sq. Meter` | **Transformed** into `total_sqft` (float) |
| `bath` | `float64` | Count | Number of attached and common bathrooms. | `1.0`, `2.0`, `3.0`, `4.0` | **Yes** (Numerical) |
| `balcony` | `float64` | Count | Number of balconies. | `0.0`, `1.0`, `2.0`, `3.0` | **Yes** (Numerical) |
| `price` | `float64` | ₹ Lakhs INR | Total transaction/market valuation price in Indian Rupees (1 Lakh = 100,000 INR). | `39.07`, `120.0`, `62.0` | **Target Variable** ($y$) |

---

## 3. Cleaned & Processed Features (Pipeline Ingestion)

| Feature Name | Cleaned Type | Unit | Transformation / Cleaning Logic | Used for Prediction? |
| :--- | :--- | :--- | :--- | :--- |
| `location` | `string` | N/A | Stripped whitespace, title cased, low-frequency localities (<10 transactions) aggregated into `'Other'`. | **Yes** |
| `total_sqft` | `float64` | Sq. Feet | Parsed ranges via midpoint $(low + high)/2$, converted metric units to sqft ($1\text{ m}^2 = 10.7639\text{ ft}^2$). | **Yes** |
| `bhk` | `int64` | Count | Regex extracted integer number of bedrooms from raw `size` strings. | **Yes** |
| `bath` | `int64` | Count | Imputed missing values using grouped BHK median; checked for domain consistency ($bath \le bhk + 2$). | **Yes** |
| `balcony` | `int64` | Count | Imputed missing values using grouped BHK median; clipped to non-negative counts. | **Yes** |
| `area_type` | `string` | N/A | Missing filled with mode (`Super built-up Area`), whitespace stripped. | **Yes** |
| `availability_status` | `string` | N/A | Standardized binary categorical: `'Ready To Move'` vs. `'Under Construction'`. | **Yes** |
| `price` | `float64` | ₹ Lakhs INR | Verified strictly positive ($price > 0$), domain anomalies ($price / sqft$) filtered. | **Target ($y$)** |

---

## 4. Derived Features for Exploratory Data Analysis (EDA)

> [!CAUTION]
> **Data Leakage Safeguard**: 
> Features derived from the target variable `price` (such as `price_per_sqft`) are strictly isolated for **Exploratory Data Analysis (EDA) and Market Analytics only**. 
> They are **NEVER** supplied as input features to machine learning models during training or inference.

| Feature Name | Type | Formula / Origin | Purpose | Included in ML Inputs? |
| :--- | :--- | :--- | :--- | :--- |
| `price_per_sqft` | `float64` | $\frac{\text{price} \times 100,000}{\text{total\_sqft}}$ | Locality valuation ranking, market hotspot detection, outlier visualization. | **NO (Target Leakage)** |
| `sqft_per_bhk` | `float64` | $\frac{\text{total\_sqft}}{\text{bhk}}$ | Spatial density, luxury indicator, domain sanity checks ($\ge 200\text{ sqft/BHK}$). | Optional Feature |
| `bhk_bath_ratio` | `float64` | $\frac{\text{bhk}}{\text{bath}}$ | Room convenience ratio and luxury distribution. | Optional Feature |


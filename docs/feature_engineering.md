# Feature Engineering & Preprocessing

**Dataset:** 12,395 Bengaluru residential properties  
**Target:** `price` (₹ Lakhs)  
**Split:** 80/20, random state 42  
**Model-ready dimensions:** 224 features after rare-locality handling and one-hot encoding

## Input features

Numerical:
- `total_sqft`
- `bath`
- `balcony`
- `bhk`

Categorical:
- `location`
- `area_type`
- `availability_status`

## Engineered features

- `total_rooms = bhk + bath`
- `sqft_per_bhk = total_sqft / max(bhk, 1)`
- `bath_per_bhk = bath / max(bhk, 1)`
- `sqft_per_room = total_sqft / max(total_rooms, 1)`
- `bhk_bath_diff = bath - bhk`

## Leakage prevention

The following are blocked from model inputs:

- `price`
- `price_lakhs`
- `price_per_sqft`
- `target`
- `target_price`
- `target_encoded_price`

`price_per_sqft` remains an EDA/market-analysis metric only.

## Locality handling

Location text is normalized using whitespace and title-case normalization. Localities with fewer than 10 training observations are grouped into `Other`. This grouping is learned from training data only. `OneHotEncoder(handle_unknown="ignore")` prevents inference failures for unseen localities.

## Numerical preprocessing

Missing numerical values are median-imputed. `RobustScaler` is used to reduce sensitivity to the long-tailed area distribution.

## Target

The model trains on:

`log1p(price)`

Predictions are inverse-transformed using:

`expm1(prediction)`

All user-facing results are reported in ₹ Lakhs.

## Train/test isolation

The test set remains untouched during preprocessing and tuning. The preprocessing transformer is fitted only on training data.

## Reproducibility

`random_state=42` is used wherever supported.

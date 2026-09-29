# SmartHouse AI — Final Project Report

## 1. Problem statement

Estimate the market price of a residential property in Bengaluru from observable property characteristics while providing transparent model explanations and market context.

## 2. Dataset

- 12,395 cleaned records
- 251 Bengaluru localities
- Target: `price` in ₹ Lakhs
- Input variables: total square footage, BHK, bathrooms, balconies, locality, area type, and availability status.

The raw target is strongly right-skewed, so the primary training target uses `log1p(price)`.

## 3. Data engineering

The pipeline preserves raw data, validates schema and impossible values, handles missing values, normalizes categorical text, and groups rare localities without target-derived statistics.

## 4. Feature engineering

Five leakage-safe domain features are generated:

1. `total_rooms`
2. `sqft_per_bhk`
3. `bath_per_bhk`
4. `sqft_per_room`
5. `bhk_bath_diff`

`price_per_sqft` is intentionally excluded from prediction.

## 5. Model development

Five regressors were benchmarked:

| Model | MAE ₹L | RMSE ₹L | R² |
|---|---:|---:|---:|
| XGBoost | 35.09 | 113.98 | 0.630 |
| LightGBM | 35.83 | 114.46 | 0.627 |
| Random Forest | 37.47 | 117.66 | 0.606 |
| Gradient Boosting | 39.60 | 139.54 | 0.446 |
| Ridge | 52.55 | 267.03 | -1.030 |

XGBoost was tuned using a validation split inside the training data and then retrained on the full training partition.

## 6. Final held-out evaluation

The final XGBoost model was evaluated once on the untouched 20% test set:

- **MAE:** ₹34.94 Lakhs
- **RMSE:** ₹113.33 Lakhs
- **R²:** 0.634
- **Median absolute error:** ₹11.77 Lakhs
- **MAPE:** 23.20%
- **Within ±10%:** 32.39%
- **Within ±20%:** 58.13%
- **Within ±30%:** 75.19%

The large gap between RMSE and median absolute error is expected because the dataset contains legitimate high-value properties and a long right tail.

## 7. Explainable AI

SHAP TreeExplainer is used with the final XGBoost model. Global analysis identifies total square footage as the strongest model feature, followed by engineered room-scale features and locality/area-type signals.

SHAP values are interpreted as model contributions, not causal effects.

## 8. Uncertainty

Five-fold out-of-fold predictions on the training set produce an empirical 90th-percentile absolute residual of approximately **₹64.26 Lakhs**. The dashboard displays this as an empirical error band.

It should not be described as a guaranteed confidence interval.

## 9. Application

The project exposes:

- `POST /predict`
- `GET /health`
- `GET /model-info`
- `GET /localities`
- `GET /market-summary`
- browser dashboard at `/dashboard`
- Swagger API documentation at `/docs`

## 10. Quality assurance

The final repository contains automated tests for:

- data cleaning
- data validation
- EDA
- feature engineering
- leakage protection
- preprocessing
- target transformation
- metrics

The current suite passes **38 tests**.

## 11. Limitations and future work

The system is deliberately Bengaluru-focused. Future work could add:

- temporal property-price trends
- richer geospatial coordinates
- amenities and distance-to-metro features from verified data
- image-based property quality signals
- calibrated prediction intervals
- city-specific models for Chennai, Hyderabad, Mumbai, etc.
- production authentication, monitoring, and model drift detection

## 12. Academic value

The project demonstrates the complete ML lifecycle rather than only model training:

**data → validation → EDA → feature engineering → leakage prevention → benchmarking → tuning → evaluation → explainability → uncertainty → API → UI → testing**

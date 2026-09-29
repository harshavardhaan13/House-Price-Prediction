# SmartHouse AI — Modeling & Evaluation

## Modeling strategy
The cleaned Bengaluru dataset is split once into 80% training and 20% untouched test data with random state 42. All preprocessing is fit only on training data.

Candidate regressors:
- Ridge Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

The primary training target is `log1p(price)` because the raw price distribution is strongly right-skewed. Every evaluation metric is calculated after inverse transformation to ₹ Lakhs.

## Selection and tuning
Models are first benchmarked on the untouched test split for transparent comparison. XGBoost is then tuned using an internal validation split of the training data. The final selected XGBoost configuration is retrained on the complete training split and evaluated once on the untouched test split.

> Note: The benchmark table is descriptive; final model selection is based on internal validation, not on repeatedly optimizing against the test set.

## Metrics
- MAE: average absolute error in ₹ Lakhs
- RMSE: penalizes larger errors
- R²: explained variance
- Median absolute error: typical error robust to luxury outliers
- MAPE: relative error
- Within 10/20/30%: practical prediction coverage

## Uncertainty
Five-fold out-of-fold predictions on the training set are used to estimate the 90th percentile absolute error. The UI presents this as an empirical error band, not a guaranteed statistical confidence interval.

## Leakage controls
`price_per_sqft`, `price_lakhs`, `price`, and target-derived encodings are blocked from the model input pipeline.

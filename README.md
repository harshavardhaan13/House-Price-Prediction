# SmartHouse AI — Explainable Bengaluru House Price Prediction

SmartHouse AI is an end-to-end machine-learning application for estimating **Bengaluru residential property prices in ₹ Lakhs**. It combines a leakage-safe preprocessing pipeline, multiple regression models, XGBoost tuning, SHAP explanations, an empirical error band, a FastAPI backend, and an interactive dashboard.

## What is implemented

- 12,395 cleaned Bengaluru property records across 251 localities
- Leakage protection for `price`, `price_per_sqft`, and target-derived features
- Feature engineering: total rooms, sqft/BHK, bath/BHK, sqft/room, BHK-bath difference
- 80/20 reproducible train/test split (`random_state=42`)
- Model benchmark: Ridge, Random Forest, Gradient Boosting, XGBoost, LightGBM
- XGBoost internal validation tuning
- Final XGBoost model trained on `log1p(price)`
- Original-scale evaluation in ₹ Lakhs
- SHAP local/global explainability
- OOF residual-based empirical 90% error band
- FastAPI REST API with Swagger docs
- Responsive browser dashboard
- Streamlit dashboard alternative
- Automated pytest suite

## Final model

The selected model is **XGBoost**.

Held-out test results:

| Metric | Result |
|---|---:|
| MAE | ₹34.94 L |
| RMSE | ₹113.33 L |
| R² | 0.634 |
| Median absolute error | ₹11.77 L |
| MAPE | 23.20% |
| Within ±10% | 32.39% |
| Within ±20% | 58.13% |
| Within ±30% | 75.19% |

The dataset contains a long luxury-property tail, so RMSE is much larger than the median error. The model is therefore reported with multiple metrics rather than a single accuracy percentage.

## Run locally

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Train/rebuild the model

```bash
python -m src.evaluation.evaluate
```

This creates artifacts under `models/`.

### 3. Start the API + browser dashboard

```bash
uvicorn backend.main:app --reload
```

Then open:

- Dashboard: `http://127.0.0.1:8000/dashboard`
- API docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

### 4. Optional Streamlit UI

```bash
streamlit run frontend/app.py
```

## Project structure

```text
SmartHouse-AI/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   ├── explainability/
│   ├── database/
│   └── utils/
├── backend/
├── frontend/
├── models/
├── tests/
├── notebooks/
├── visualizations/
└── docs/
```

## Important modeling safeguards

`price_per_sqft` is useful for market analysis but is **not a model input**, because it is calculated from the target price. Preprocessing is fitted only on the training split. Unseen localities are handled through `OneHotEncoder(handle_unknown="ignore")`, and rare localities are grouped without using target information.

## Limitations

This is a Bengaluru-focused dataset, not a pan-India valuation model. Predictions should be treated as estimates rather than appraisals. The empirical error band is derived from out-of-fold residuals and is not a guaranteed confidence interval.

## Documentation

See:

- `docs/feature_engineering.md`
- `docs/modeling.md`
- `docs/explainability.md`
- `docs/deployment.md`
- `docs/final_project_report.md`

## Prediction disclaimer

SmartHouse AI provides a historical-data-based estimate from the Bengaluru dataset. It is an educational/analytical ML system, not a live property appraisal, investment recommendation, or guaranteed market value.

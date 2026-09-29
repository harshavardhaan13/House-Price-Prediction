# Running SmartHouse AI

## Install
```bash
python -m pip install -r requirements.txt
```

## Train
```bash
python -m src.evaluation.evaluate
```

This creates model artifacts under `models/`.

## API
```bash
uvicorn backend.main:app --reload
```
Open `/docs` for interactive Swagger documentation.

## Dashboard
```bash
streamlit run frontend/app.py
```

The dashboard provides property inputs, price prediction, an empirical 90% error band, and SHAP explanation.

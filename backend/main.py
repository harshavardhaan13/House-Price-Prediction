"""SmartHouse AI FastAPI service."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, RedirectResponse
from joblib import load
from src.utils.config import MODELS_DIR, CLEANED_DATA_PATH
from src.features.engineering import inverse_transform_target
from src.explainability.shap_explainer import HousePriceExplainer

ARTIFACT_PATH=MODELS_DIR/"smarthouse_model.joblib"
app=FastAPI(title="SmartHouse AI API",version="1.0.0",
            description="Explainable Bengaluru residential property valuation API")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,
                   allow_methods=["*"],allow_headers=["*"])

class PropertyInput(BaseModel):
    location: str = Field(..., examples=["Whitefield"])
    total_sqft: float = Field(..., gt=0, examples=[1200])
    bath: float = Field(..., ge=0, examples=[2])
    balcony: float = Field(..., ge=0, examples=[1])
    bhk: float = Field(..., gt=0, examples=[2])
    area_type: str = Field(..., examples=["Super built-up  Area"])
    availability_status: str = Field(..., examples=["Ready To Move"])

def artifact():
    if not ARTIFACT_PATH.exists():
        raise HTTPException(503,"Model artifact not found. Run: python -m src.evaluation.evaluate")
    return load(ARTIFACT_PATH)

@app.get("/health")
def health():
    return {"status":"ok","model_ready":ARTIFACT_PATH.exists()}

@app.get("/model-info")
def model_info():
    a=artifact()
    return a["metadata"]

@app.get("/localities")
def localities():
    d=pd.read_csv(CLEANED_DATA_PATH)
    return {"count":int(d.location.nunique()),"localities":sorted(d.location.dropna().unique().tolist())}

@app.get("/market-summary")
def market_summary():
    d=pd.read_csv(CLEANED_DATA_PATH)
    by=d.groupby("location").agg(properties=("price","size"),median_price_lakhs=("price","median"),
                                 mean_price_lakhs=("price","mean")).reset_index()
    return {"records":len(d),"median_price_lakhs":float(d.price.median()),
            "mean_price_lakhs":float(d.price.mean()),
            "top_localities":by.sort_values("median_price_lakhs",ascending=False).head(15).to_dict("records")}

@app.post("/predict")
def predict(p:PropertyInput):
    a=artifact()
    raw=pd.DataFrame([p.model_dump()])
    model=a["model"]; pipe=a["feature_pipeline"]
    transformed=pipe.transform(raw)
    log_pred=float(model.predict(transformed)[0])
    pred=float(inverse_transform_target(log_pred,"log1p"))
    q=float(a.get("interval_q90_lakhs",0))
    explanation=HousePriceExplainer(a).explain(raw,top_n=8)
    return {"predicted_price_lakhs":round(pred,2),
            "predicted_price_crore":round(pred/100,2),
            "interval_90_lakhs":[round(max(0,pred-q),2),round(pred+q,2)],
            "interval_label":"Empirical historical-error range (not a formal confidence interval)",
            "explanation":explanation}

@app.get("/dashboard", include_in_schema=False)
def dashboard():
    return FileResponse(Path(__file__).resolve().parents[1] / "frontend" / "index.html")

@app.get("/")
def root():
    return RedirectResponse(url="/dashboard", status_code=307)

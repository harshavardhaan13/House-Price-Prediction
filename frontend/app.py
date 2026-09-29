"""Streamlit UI for SmartHouse AI. Run with: streamlit run frontend/app.py"""
import sys, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import pandas as pd
import streamlit as st
from joblib import load
from src.utils.config import MODELS_DIR, CLEANED_DATA_PATH
from src.features.engineering import inverse_transform_target
from src.explainability.shap_explainer import HousePriceExplainer

st.set_page_config(page_title="SmartHouse AI",page_icon="🏠",layout="wide")
st.title("🏠 SmartHouse AI")
st.caption("Explainable Residential Property Valuation")

ART=MODELS_DIR/"smarthouse_model.joblib"
if not ART.exists():
    st.error("Model is not trained yet. Run `python -m src.evaluation.evaluate`.")
    st.stop()
artifact=load(ART)
df=pd.read_csv(CLEANED_DATA_PATH)
model=artifact["model"]; pipe=artifact["feature_pipeline"]
explainer=HousePriceExplainer(artifact)

with st.sidebar:
    st.header("Property Details")
    location=st.selectbox("Locality",sorted(df.location.unique()))
    area=st.number_input("Total area (sq.ft)",min_value=200.0,max_value=20000.0,value=1200.0,step=50.0)
    bhk=st.number_input("BHK",min_value=1.0,max_value=12.0,value=2.0,step=1.0)
    bath=st.number_input("Bathrooms",min_value=1.0,max_value=12.0,value=2.0,step=1.0)
    balcony=st.number_input("Balconies",min_value=0.0,max_value=3.0,value=1.0,step=1.0)
    area_type=st.selectbox("Area type",sorted(df.area_type.unique()))
    availability=st.selectbox("Availability",sorted(df.availability_status.unique()))

raw=pd.DataFrame([{"location":location,"total_sqft":area,"bath":bath,"balcony":balcony,
                   "bhk":bhk,"area_type":area_type,"availability_status":availability}])
pred=float(inverse_transform_target(model.predict(pipe.transform(raw))[0],"log1p"))
q=float(artifact["interval_q90_lakhs"])
c1,c2,c3=st.columns(3)
c1.metric("Estimated price",f"₹{pred:.1f} L")
c2.metric("Approx. crores",f"₹{pred/100:.2f} Cr")
c3.metric("Empirical error band",f"±₹{q:.1f} L")
st.divider()
st.subheader("Why this prediction?")
exp=explainer.explain(raw,top_n=8)
st.dataframe(pd.DataFrame(exp)[["feature","impact","shap_value_log_price","magnitude"]],
             use_container_width=True,hide_index=True)
st.subheader("Bengaluru Market Snapshot")
st.metric("Properties in dataset",f"{len(df):,}")
st.metric("Median market price",f"₹{df.price.median():.1f} L")
st.metric("Localities",f"{df.location.nunique()}")
st.caption("The model never uses price_per_sqft or other target-derived variables as prediction inputs.")

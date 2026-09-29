import pandas as pd

from src.explainability.shap_explainer import HousePriceExplainer
from src.utils.config import MODELS_DIR
from joblib import load


def test_shap_explanation_omits_inactive_one_hot_categories():
    artifact = load(MODELS_DIR / "smarthouse_model.joblib")
    explainer = HousePriceExplainer(artifact)
    raw = pd.DataFrame([{
        "location": "Whitefield",
        "total_sqft": 1500.0,
        "bath": 3.0,
        "balcony": 2.0,
        "bhk": 3.0,
        "area_type": "Super built-up Area",
        "availability_status": "Ready To Move",
    }])
    explanation = explainer.explain(raw, top_n=20)
    names = [item["raw_feature"] for item in explanation]
    assert not any("cat__area_type_Plot Area" == n for n in names)
    assert not any("cat__availability_status_Under Construction" == n for n in names)
    assert all(item["is_active"] for item in explanation)


def test_friendly_names_are_user_facing():
    assert HousePriceExplainer._friendly_name("cat__area_type_Super Built-up Area") == "Area type: Super Built-up Area"
    assert HousePriceExplainer._friendly_name("cat__location_Whitefield") == "Locality: Whitefield"
    assert HousePriceExplainer._friendly_name("num__total_sqft") == "Property size"

"""Model factory and registry for SmartHouse AI."""
from typing import Dict
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def get_model_registry(random_state: int = 42) -> Dict[str, object]:
    return {
        "Ridge": Ridge(alpha=10.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, max_features=0.8,
            n_jobs=-1, random_state=random_state
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.04, max_depth=3,
            min_samples_leaf=5, loss="huber", random_state=random_state
        ),
        "XGBoost": XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
            reg_lambda=2.0, objective="reg:squarederror",
            n_jobs=-1, random_state=random_state
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=600, num_leaves=31, learning_rate=0.04,
            min_child_samples=20, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=1.0, random_state=random_state, n_jobs=-1, verbosity=-1
        ),
    }

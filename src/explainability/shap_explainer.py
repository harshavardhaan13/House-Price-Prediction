"""SHAP-based local and global explanations for the trained XGBoost model.

The local explanation intentionally reports only *active* model features.  This
prevents one-hot encoded categories that are zero for the current property (for
example ``Plot Area`` when the user selected ``Super built-up Area``) from being
shown as if they were properties of the prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap


class HousePriceExplainer:
    def __init__(self, artifact):
        self.model = artifact["model"]
        self.pipeline = artifact["feature_pipeline"]
        self.feature_names = list(
            self.pipeline.named_steps["preprocessor"].get_feature_names_out()
        )
        self.explainer = shap.TreeExplainer(self.model)

    def explain(self, raw_df: pd.DataFrame, top_n: int = 8):
        """Return top SHAP contributions from features active for this property.

        For one-hot categorical variables, inactive categories are omitted. This
        keeps the explanation aligned with the property the user actually entered.
        SHAP values remain in model log-price space and are explicitly labelled as
        model contributions, not causal effects.
        """
        transformed = self.pipeline.transform(raw_df)
        values = self.explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[0]
        vals = np.asarray(values)[0]
        row = np.asarray(transformed)[0]

        candidates = []
        for i, (name, shap_value, feature_value) in enumerate(
            zip(self.feature_names, vals, row)
        ):
            # One-hot encoded categorical features are meaningful only when active.
            # Numerical features should always be eligible for explanation.
            is_categorical = name.startswith("cat__")
            if is_categorical and not np.isclose(feature_value, 1.0):
                continue
            if np.isclose(shap_value, 0.0):
                continue
            candidates.append((i, float(shap_value)))

        candidates.sort(key=lambda item: abs(item[1]), reverse=True)
        candidates = candidates[:top_n]

        results = []
        for i, shap_value in candidates:
            raw_name = self.feature_names[i]
            results.append(
                {
                    "feature": self._friendly_name(raw_name),
                    "raw_feature": raw_name,
                    "shap_value_log_price": shap_value,
                    "impact": "increases" if shap_value > 0 else "decreases",
                    "magnitude": abs(shap_value),
                    "is_active": True,
                }
            )
        return results

    @staticmethod
    def _friendly_name(name: str) -> str:
        name = name.replace("num__", "").replace("cat__", "")
        labels = {
            "total_sqft": "Property size",
            "total_rooms": "Total rooms",
            "sqft_per_bhk": "Area per BHK",
            "bath_per_bhk": "Bathrooms per BHK",
            "sqft_per_room": "Area per room",
            "bhk_bath_diff": "BHK/bathroom difference",
            "bhk": "BHK",
            "bath": "Bathrooms",
            "balcony": "Balconies",
        }
        if name in labels:
            return labels[name]
        if name.startswith("area_type_"):
            return f"Area type: {name[len('area_type_'):]}"
        if name.startswith("location_"):
            return f"Locality: {name[len('location_'):]}"
        if name.startswith("availability_status_"):
            return f"Availability: {name[len('availability_status_'):]}"
        return name.replace("_", " ").title()

    def global_importance(self, transformed_X, top_n=20):
        values = self.explainer.shap_values(transformed_X)
        if isinstance(values, list):
            values = values[0]
        imp = np.mean(np.abs(values), axis=0)
        idx = np.argsort(imp)[::-1][:top_n]
        return [
            {
                "feature": self._friendly_name(self.feature_names[i]),
                "raw_feature": self.feature_names[i],
                "mean_abs_shap": float(imp[i]),
            }
            for i in idx
        ]

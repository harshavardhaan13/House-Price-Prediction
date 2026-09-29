# Explainable AI

SmartHouse AI uses SHAP with the final XGBoost model.

For an individual property, the system returns the strongest feature contributions in model (log-price) space. Positive SHAP values push the estimate upward; negative values push it downward.

The interface intentionally displays explanations as feature impacts rather than claiming causal effects. A SHAP contribution explains the model's prediction, not a guaranteed real-world causal relationship.

## Local explanation display policy

The dashboard shows only active one-hot categorical features for the property being evaluated. For example, a Super Built-up Area prediction will not display inactive categories such as Plot Area as an explanation. SHAP values are reported in the model log-price space and describe how features move the model output relative to its baseline; they are not causal effects.

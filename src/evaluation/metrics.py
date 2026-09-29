"""Evaluation metrics on the original ₹ Lakhs price scale."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0)
    err = np.abs(y_true - y_pred)
    nonzero = np.abs(y_true) > 1e-9
    return {
        "MAE_lakhs": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_lakhs": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MedianAE_lakhs": float(median_absolute_error(y_true, y_pred)),
        "MAPE_percent": float(np.mean(err[nonzero] / np.abs(y_true[nonzero])) * 100),
        "Within_10pct_percent": float(np.mean(err <= 0.10 * np.abs(y_true)) * 100),
        "Within_20pct_percent": float(np.mean(err <= 0.20 * np.abs(y_true)) * 100),
        "Within_30pct_percent": float(np.mean(err <= 0.30 * np.abs(y_true)) * 100),
        "Mean_Error_lakhs": float(np.mean(y_pred - y_true)),
    }

from src.evaluation.metrics import regression_metrics
def test_metrics_keys():
    m=regression_metrics([10,20,30],[10,20,30])
    assert m["RMSE_lakhs"] == 0.0
    assert m["MAE_lakhs"] == 0.0
    assert m["R2"] == 1.0

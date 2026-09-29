"""Model training, benchmarking, tuning, evaluation and artifact creation."""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from joblib import dump

from src.features.pipeline import build_feature_pipeline
from src.features.engineering import transform_target, inverse_transform_target
from src.models.model_registry import get_model_registry
from src.evaluation.metrics import regression_metrics
from src.utils.config import CLEANED_DATA_PATH, MODELS_DIR, RANDOM_SEED, TARGET_COLUMN, TEST_DATA_PATH, TRAIN_DATA_PATH

def _fit_predict(model, Xtr, ytr, Xva, log_target=True):
    yfit = transform_target(ytr, "log1p") if log_target else ytr
    model.fit(Xtr, yfit)
    pred = model.predict(Xva)
    return inverse_transform_target(pred, "log1p") if log_target else np.maximum(pred, 0)

def benchmark_models(train_df, test_df, log_target=True):
    Xtr_raw = train_df.drop(columns=[TARGET_COLUMN])
    Xte_raw = test_df.drop(columns=[TARGET_COLUMN])
    ytr, yte = train_df[TARGET_COLUMN].values, test_df[TARGET_COLUMN].values
    pipe = build_feature_pipeline(scaler_type="robust")
    Xtr = pipe.fit_transform(Xtr_raw)
    Xte = pipe.transform(Xte_raw)
    results = []
    for name, model in get_model_registry(RANDOM_SEED).items():
        start = time.perf_counter()
        pred = _fit_predict(model, Xtr, ytr, Xte, log_target)
        row = {"model": name, "target_transform": "log1p" if log_target else "identity",
               "fit_seconds": round(time.perf_counter()-start, 3)}
        row.update(regression_metrics(yte, pred))
        results.append(row)
    return pd.DataFrame(results).sort_values(["MAE_lakhs", "RMSE_lakhs"]).reset_index(drop=True)

def tune_xgboost(train_df, log_target=True):
    """Small leakage-safe validation search; the final test set remains untouched."""
    inner_train, inner_val = train_test_split(train_df, test_size=0.20, random_state=RANDOM_SEED)
    candidates = [
        dict(n_estimators=400, max_depth=5, learning_rate=0.05, min_child_weight=3),
        dict(n_estimators=600, max_depth=6, learning_rate=0.04, min_child_weight=3),
        dict(n_estimators=700, max_depth=7, learning_rate=0.035, min_child_weight=5),
        dict(n_estimators=800, max_depth=5, learning_rate=0.035, min_child_weight=2),
    ]
    rows=[]
    best=None
    for p in candidates:
        pipe=build_feature_pipeline()
        Xtr=pipe.fit_transform(inner_train.drop(columns=[TARGET_COLUMN]))
        Xv=pipe.transform(inner_val.drop(columns=[TARGET_COLUMN]))
        model=get_model_registry()[ "XGBoost" ]
        model.set_params(**p)
        pred=_fit_predict(model, Xtr, inner_train[TARGET_COLUMN].values,
                          Xv, log_target)
        score=mean_absolute_error(inner_val[TARGET_COLUMN].values,pred)
        rows.append({**p,"validation_MAE_lakhs":float(score)})
        if best is None or score < best[0]:
            best=(score,p)
    return best[1], pd.DataFrame(rows).sort_values("validation_MAE_lakhs")

def train_final():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(CLEANED_DATA_PATH)
    train_df, test_df=train_test_split(df,test_size=0.20,random_state=RANDOM_SEED,shuffle=True)
    train_df=train_df.reset_index(drop=True); test_df=test_df.reset_index(drop=True)
    train_df.to_csv(TRAIN_DATA_PATH,index=False); test_df.to_csv(TEST_DATA_PATH,index=False)

    benchmark=benchmark_models(train_df,test_df,log_target=True)
    benchmark.to_json(MODELS_DIR/"model_comparison.json",orient="records",indent=2)

    best_params,tuning=tune_xgboost(train_df,True)
    tuning.to_csv(MODELS_DIR/"xgboost_tuning.csv",index=False)

    # Fit final pipeline/model on all training data only.
    feature_pipeline=build_feature_pipeline()
    Xtr=feature_pipeline.fit_transform(train_df.drop(columns=[TARGET_COLUMN]))
    Xte=feature_pipeline.transform(test_df.drop(columns=[TARGET_COLUMN]))
    final_model=get_model_registry()[ "XGBoost" ]
    final_model.set_params(**best_params)
    final_model.fit(Xtr,transform_target(train_df[TARGET_COLUMN].values,"log1p"))
    pred=np.maximum(inverse_transform_target(final_model.predict(Xte),"log1p"),0)
    test_metrics=regression_metrics(test_df[TARGET_COLUMN].values,pred)

    # Out-of-fold residual calibration for an honest uncertainty band.
    kf=KFold(n_splits=5,shuffle=True,random_state=RANDOM_SEED)
    oof=np.zeros(len(train_df))
    for tr_idx,va_idx in kf.split(train_df):
        fold_pipe=build_feature_pipeline()
        fold_model=get_model_registry()[ "XGBoost" ]
        fold_model.set_params(**best_params)
        Xf=fold_pipe.fit_transform(train_df.iloc[tr_idx].drop(columns=[TARGET_COLUMN]))
        Xv=fold_pipe.transform(train_df.iloc[va_idx].drop(columns=[TARGET_COLUMN]))
        fold_model.fit(Xf,transform_target(train_df.iloc[tr_idx][TARGET_COLUMN].values,"log1p"))
        oof[va_idx]=inverse_transform_target(fold_model.predict(Xv),"log1p")
    abs_res=np.abs(train_df[TARGET_COLUMN].values-oof)
    interval_q=float(np.quantile(abs_res,0.90))

    metadata={
        "project":"SmartHouse AI",
        "scope":"Bengaluru residential property valuation",
        "target":"price",
        "target_unit":"₹ Lakhs",
        "target_transform":"log1p",
        "random_state":RANDOM_SEED,
        "train_rows":len(train_df),"test_rows":len(test_df),
        "best_model":"XGBoost","best_params":best_params,
        "test_metrics":test_metrics,
        "benchmark":benchmark.to_dict(orient="records"),
        "oof_absolute_error_q90_lakhs":interval_q,
        "feature_count":int(Xtr.shape[1]),
        "localities":int(df["location"].nunique()),
    }
    dump({"model":final_model,"feature_pipeline":feature_pipeline,
          "target_transform":"log1p","interval_q90_lakhs":interval_q,
          "metadata":metadata},MODELS_DIR/"smarthouse_model.joblib")
    dump(feature_pipeline,MODELS_DIR/"preprocessor.joblib")
    dump(final_model,MODELS_DIR/"best_model.joblib")
    (MODELS_DIR/"metadata.json").write_text(json.dumps(metadata,indent=2,default=float),encoding="utf-8")

    pd.DataFrame({"actual_price":test_df[TARGET_COLUMN].values,
                  "predicted_price":pred,
                  "absolute_error":np.abs(test_df[TARGET_COLUMN].values-pred)}
                ).to_csv(MODELS_DIR/"test_predictions.csv",index=False)
    return metadata

if __name__=="__main__":
    m=train_final()
    print(json.dumps(m["test_metrics"],indent=2))

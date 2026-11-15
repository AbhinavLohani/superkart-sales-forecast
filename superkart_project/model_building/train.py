"""Model training with hyper-parameter tuning and MLflow experiment tracking."""
import json
import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "SuperKart-Sales-Forecast"
MODEL_DIR = "superkart_project/deployment"
MODEL_PATH = f"{MODEL_DIR}/superkart_model.joblib"
METRICS_PATH = f"{MODEL_DIR}/model_metrics.json"
RANDOM_STATE = 42
CV_FOLDS = 3

NUMERIC = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Establishment_Year"]
CATEGORICAL = ["Product_Sugar_Content", "Product_Type", "Store_Size",
               "Store_Location_City_Type", "Store_Type", "Product_Category"]


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_pipeline(model) -> Pipeline:
    preprocessor = make_column_transformer(
        (StandardScaler(), NUMERIC),
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    )
    return Pipeline([("prep", preprocessor), ("model", model)])


# model candidates and their hyper-parameter grids
CANDIDATES = {
    "RandomForest": (
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {"model__n_estimators": [100, 200], "model__max_depth": [6, 10, 14], "model__min_samples_leaf": [2, 5]},
    ),
    "GradientBoosting": (
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1], "model__max_depth": [3, 4]},
    ),
    "XGBoost": (
        xgb.XGBRegressor(random_state=RANDOM_STATE, subsample=0.8, colsample_bytree=0.8, n_jobs=-1),
        {"model__n_estimators": [200, 400], "model__learning_rate": [0.03, 0.1], "model__max_depth": [3, 5]},
    ),
}


def clean_params(params: dict) -> dict:
    return {k.replace("model__", ""): v for k, v in params.items()}


def main() -> None:
    Xtrain, Xtest = pd.read_csv("Xtrain.csv"), pd.read_csv("Xtest.csv")
    ytrain, ytest = pd.read_csv("ytrain.csv").squeeze("columns"), pd.read_csv("ytest.csv").squeeze("columns")
    print(f"[train] train {Xtrain.shape}, test {Xtest.shape}")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"[train] MLflow tracking uri: {TRACKING_URI}")

    results = {}
    with mlflow.start_run(run_name="tuning-all-models"):
        for name, (model, grid) in CANDIDATES.items():
            search = GridSearchCV(
                build_pipeline(model), grid, cv=CV_FOLDS, n_jobs=-1, refit="rmse",
                scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"},
            )
            search.fit(Xtrain, ytrain)

            with mlflow.start_run(run_name=name, nested=True):
                # log EVERY tuned parameter combination as its own child run
                cvr = search.cv_results_
                for i, params in enumerate(cvr["params"]):
                    with mlflow.start_run(run_name=f"{name}-cfg{i:02d}", nested=True):
                        mlflow.log_params(clean_params(params))
                        mlflow.log_metric("cv_r2", float(cvr["mean_test_r2"][i]))
                        mlflow.log_metric("cv_rmse", float(-cvr["mean_test_rmse"][i]))

                best = search.best_estimator_
                train_m = regression_metrics(ytrain, best.predict(Xtrain))
                test_m = regression_metrics(ytest, best.predict(Xtest))
                cv_rmse = float(-search.best_score_)

                mlflow.log_params({"model": name, **clean_params(search.best_params_)})
                mlflow.log_metric("cv_rmse", cv_rmse)
                mlflow.log_metrics({f"train_{k}": v for k, v in train_m.items()})
                mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})

            results[name] = {"best_params": clean_params(search.best_params_), "cv_rmse": cv_rmse,
                             "train": train_m, "test": test_m, "estimator": best}
            print(f"[train] {name:<17} cv_rmse={cv_rmse:8.2f} | test rmse={test_m['rmse']:8.2f} "
                  f"mae={test_m['mae']:8.2f} r2={test_m['r2']:.4f} | best={clean_params(search.best_params_)}")

        # select the best model on cross-validated RMSE (test data is NOT used for selection)
        best_name = min(results, key=lambda n: results[n]["cv_rmse"])
        best = results[best_name]
        print(f"[train] >>> best model: {best_name}")

        mlflow.log_param("best_model", best_name)
        mlflow.log_metrics({f"best_test_{k}": v for k, v in best["test"].items()})
        mlflow.sklearn.log_model(best["estimator"], name="best_model")

    # save the best pipeline so the workflow can commit it into the repo
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best["estimator"], MODEL_PATH, compress=3)   # compressed: keeps the committed file small
    summary = {
        "best_model": best_name,
        "best_params": best["best_params"],
        "train_metrics": best["train"],
        "test_metrics": best["test"],
        "all_models": {n: {"cv_rmse": r["cv_rmse"], "test_rmse": r["test"]["rmse"], "test_r2": r["test"]["r2"]}
                       for n, r in results.items()},
    }
    with open(METRICS_PATH, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[train] saved model -> {MODEL_PATH} ({os.path.getsize(MODEL_PATH)/1024:.0f} KB)")
    print(f"[train] saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Train hybrid tabular models on the shared Kepler train/validation split.

Implemented routes:
- qpgp_residual: predict residual (P_label - P_qpgp)
- cnn_feature_fusion: direct regression using CNN + baseline/statistical features

Outputs for each route:
- best fitted sklearn pipeline
- predictions CSV
- summary CSV
- feature importance CSV (permutation importance on validation split)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "kepler"
MODEL_ROOT = DATA_ROOT / "models"

BASELINE_PATH = DATA_ROOT / "baseline_periods_ls_acf.csv"
OVERVIEW_PATH = DATA_ROOT / "lc_overview.csv"
LIGHTPRED_PRED_PATH = MODEL_ROOT / "lightpred_predictions.csv"
CNN_PRED_PATH = MODEL_ROOT / "cnn" / "cnn_predictions.csv"
LSTM_PRED_PATH = MODEL_ROOT / "lstm" / "lstm_predictions.csv"


def _metrics_df(df: pd.DataFrame, pred_col: str, label_col: str = "prot_label", subset: str = "val") -> pd.DataFrame:
    tmp = df[[label_col, pred_col]].replace([np.inf, -np.inf], np.nan).dropna()
    err = tmp[pred_col] - tmp[label_col]
    abs_err = err.abs()
    rel_err = abs_err / tmp[label_col].replace(0, np.nan)
    return pd.DataFrame(
        [
            {
                "subset": subset,
                "N": int(len(tmp)),
                "MAE": float(abs_err.mean()),
                "RMSE": float(np.sqrt(np.mean(err**2))),
                "MedianAbsErr": float(abs_err.median()),
                "Bias": float(err.mean()),
                "MedianBias": float(err.median()),
                "Frac(<10%)": float((rel_err < 0.10).mean()),
                "Frac(<20%)": float((rel_err < 0.20).mean()),
                "Corr": float(tmp[label_col].corr(tmp[pred_col])),
            }
        ]
    )


def _build_frame() -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE_PATH)
    overview = pd.read_csv(OVERVIEW_PATH)
    lightpred = pd.read_csv(LIGHTPRED_PRED_PATH)
    cnn = pd.read_csv(CNN_PRED_PATH)
    lstm = pd.read_csv(LSTM_PRED_PATH)

    split_map = lightpred[lightpred["split"].isin(["train", "val"])][["kic", "split"]].copy()
    if split_map["kic"].duplicated().any():
        raise RuntimeError("Duplicate KIC found in split map.")

    cnn_all = cnn[cnn["split"] == "all"][["kic", "pred_period"]].rename(columns={"pred_period": "cnn_pred"})
    lstm_all = lstm[lstm["split"] == "all"][["kic", "pred_period"]].rename(columns={"pred_period": "lstm_pred"})
    lightpred_all = lightpred[lightpred["split"] == "all"][["kic", "pred_period"]].rename(
        columns={"pred_period": "lightpred_pred"}
    )

    df = baseline.merge(split_map, on="kic", how="inner")
    df = df.merge(overview, on="kic", how="left", suffixes=("", "_ov"))
    df = df.merge(cnn_all, on="kic", how="left")
    df = df.merge(lstm_all, on="kic", how="left")
    df = df.merge(lightpred_all, on="kic", how="left")

    df["qpgp_success"] = df["qpgp_success"].astype(float)
    df["acf_available"] = df["prot_acf"].notna().astype(float)
    df["gps_available"] = df["prot_gps"].notna().astype(float)
    df["ls_qpgp_gap"] = (df["prot_ls"] - df["prot_qpgp"]).abs()
    df["gps_qpgp_gap"] = (df["prot_gps"] - df["prot_qpgp"]).abs()
    df["cnn_qpgp_gap"] = (df["cnn_pred"] - df["prot_qpgp"]).abs()
    df["cnn_ls_gap"] = (df["cnn_pred"] - df["prot_ls"]).abs()
    df["lstm_qpgp_gap"] = (df["lstm_pred"] - df["prot_qpgp"]).abs()
    df["lightpred_qpgp_gap"] = (df["lightpred_pred"] - df["prot_qpgp"]).abs()
    df["log_flux_std"] = np.log10(df["flux_std"].clip(lower=1e-6))
    df["log_flux_median"] = np.log10(df["flux_median"].clip(lower=1e-6))

    # Drop the label duplicate from overview if present.
    if "prot" in df.columns:
        df = df.drop(columns=["prot"])
    return df


def _make_candidate_search(random_state: int = 42) -> GridSearchCV:
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", Ridge()),
        ]
    )
    param_grid = [
        {"model": [Ridge()], "model__alpha": [1.0, 5.0, 10.0, 20.0]},
        {
            "model": [RandomForestRegressor(random_state=random_state, n_jobs=-1)],
            "model__n_estimators": [300, 600],
            "model__min_samples_leaf": [3, 5, 8],
            "model__max_depth": [None, 6],
        },
        {
            "model": [HistGradientBoostingRegressor(random_state=random_state)],
            "model__learning_rate": [0.03, 0.05],
            "model__max_depth": [2, 3],
            "model__max_iter": [200, 300],
            "model__min_samples_leaf": [10, 20],
        },
    ]
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )


def _fit_qpgp_residual(df: pd.DataFrame, out_dir: Path) -> None:
    features = [
        "prot_qpgp",
        "prot_ls",
        "prot_gps",
        "prot_acf",
        "q_gps",
        "qpgp_logL",
        "qpgp_success",
        "acf_available",
        "gps_available",
        "ls_qpgp_gap",
        "gps_qpgp_gap",
        "cnn_pred",
        "lightpred_pred",
        "cnn_qpgp_gap",
        "lightpred_qpgp_gap",
        "n_points",
        "time_span_days",
        "cadence_days",
        "log_flux_std",
        "log_flux_median",
        "teff",
    ]
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    X_train = train_df[features]
    y_train = train_df["prot_label"] - train_df["prot_qpgp"]
    X_val = val_df[features]

    search = _make_candidate_search()
    search.fit(X_train, y_train)

    val_pred = val_df["prot_qpgp"].to_numpy() + search.predict(X_val)
    val_out = val_df[["kic", "prot_label", "prot_qpgp"]].copy()
    val_out["pred_period"] = val_pred
    val_out["base_method"] = "QP-GP"
    val_out["abs_err"] = (val_out["pred_period"] - val_out["prot_label"]).abs()
    val_out["rel_err"] = val_out["abs_err"] / val_out["prot_label"]

    summary = _metrics_df(val_out, "pred_period")
    summary.insert(0, "method", "Hybrid-QPResidual")
    summary.insert(1, "best_estimator", type(search.best_estimator_.named_steps["model"]).__name__)
    summary.insert(2, "cv_mae", float(-search.best_score_))

    resid_importance = permutation_importance(
        search.best_estimator_,
        X_val,
        val_df["prot_label"] - val_df["prot_qpgp"],
        n_repeats=20,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )
    imp_df = pd.DataFrame(
        {
            "feature": features,
            "importance_mean": resid_importance.importances_mean,
            "importance_std": resid_importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "hybrid_qpgp_residual_model.pkl").open("wb") as f:
        pickle.dump(search, f)
    val_out.to_csv(out_dir / "hybrid_qpgp_residual_predictions.csv", index=False)
    summary.to_csv(out_dir / "hybrid_qpgp_residual_summary.csv", index=False)
    imp_df.to_csv(out_dir / "hybrid_qpgp_residual_feature_importance.csv", index=False)


def _fit_cnn_feature_fusion(df: pd.DataFrame, out_dir: Path) -> None:
    features = [
        "cnn_pred",
        "prot_qpgp",
        "prot_ls",
        "prot_gps",
        "prot_acf",
        "q_gps",
        "qpgp_logL",
        "qpgp_success",
        "acf_available",
        "gps_available",
        "ls_qpgp_gap",
        "gps_qpgp_gap",
        "cnn_qpgp_gap",
        "cnn_ls_gap",
        "lightpred_pred",
        "lstm_pred",
        "n_points",
        "time_span_days",
        "cadence_days",
        "log_flux_std",
        "log_flux_median",
        "teff",
    ]
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    X_train = train_df[features]
    y_train = train_df["prot_label"]
    X_val = val_df[features]

    search = _make_candidate_search()
    search.fit(X_train, y_train)

    val_pred = search.predict(X_val)
    val_out = val_df[["kic", "prot_label", "cnn_pred", "prot_qpgp"]].copy()
    val_out["pred_period"] = val_pred
    val_out["base_method"] = "CNN"
    val_out["abs_err"] = (val_out["pred_period"] - val_out["prot_label"]).abs()
    val_out["rel_err"] = val_out["abs_err"] / val_out["prot_label"]

    summary = _metrics_df(val_out, "pred_period")
    summary.insert(0, "method", "Hybrid-CNNFusion")
    summary.insert(1, "best_estimator", type(search.best_estimator_.named_steps["model"]).__name__)
    summary.insert(2, "cv_mae", float(-search.best_score_))

    fused_importance = permutation_importance(
        search.best_estimator_,
        X_val,
        val_df["prot_label"],
        n_repeats=20,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )
    imp_df = pd.DataFrame(
        {
            "feature": features,
            "importance_mean": fused_importance.importances_mean,
            "importance_std": fused_importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "hybrid_cnn_fusion_model.pkl").open("wb") as f:
        pickle.dump(search, f)
    val_out.to_csv(out_dir / "hybrid_cnn_fusion_predictions.csv", index=False)
    summary.to_csv(out_dir / "hybrid_cnn_fusion_summary.csv", index=False)
    imp_df.to_csv(out_dir / "hybrid_cnn_fusion_feature_importance.csv", index=False)


def main() -> None:
    df = _build_frame()
    _fit_qpgp_residual(df, MODEL_ROOT / "hybrid_qpgp_residual")
    _fit_cnn_feature_fusion(df, MODEL_ROOT / "hybrid_cnn_fusion")
    print("[INFO] Saved hybrid model outputs under data/kepler/models/")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Generate a notebook that compares all methods on the same validation split.

Outputs:
- data/kepler/models/shared_val_method_predictions.csv
- data/kepler/models/shared_val_method_summary.csv
- shared_val_model_comparison.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "kepler" / "models"
BASELINE_PATH = PROJECT_ROOT / "data" / "kepler" / "baseline_periods_ls_acf.csv"


def _load_shared_validation() -> pd.DataFrame:
    lightpred = pd.read_csv(DATA_ROOT / "lightpred_predictions.csv")
    cnn = pd.read_csv(DATA_ROOT / "cnn" / "cnn_predictions.csv")
    lstm = pd.read_csv(DATA_ROOT / "lstm" / "lstm_predictions.csv")
    baseline = pd.read_csv(BASELINE_PATH)

    lp_val = lightpred[lightpred["split"] == "val"][["kic", "true_period", "pred_period"]].copy()
    cnn_val = cnn[cnn["split"] == "val"][["kic", "pred_period"]].copy()
    lstm_val = lstm[lstm["split"] == "val"][["kic", "pred_period"]].copy()

    lp_val = lp_val.rename(columns={"true_period": "prot_label", "pred_period": "LightPred"})
    cnn_val = cnn_val.rename(columns={"pred_period": "CNN"})
    lstm_val = lstm_val.rename(columns={"pred_period": "LSTM"})

    val_kic = set(lp_val["kic"])
    base_val = baseline[baseline["kic"].isin(val_kic)].copy()
    base_val = base_val.rename(
        columns={
            "prot_ls": "LS",
            "prot_acf": "ACF",
            "prot_gps": "GPS",
            "prot_qpgp": "QP-GP",
        }
    )

    merged = lp_val.merge(cnn_val, on="kic", how="left")
    merged = merged.merge(lstm_val, on="kic", how="left")
    merged = merged.merge(base_val[["kic", "LS", "ACF", "GPS", "QP-GP"]], on="kic", how="left")
    merged = merged.sort_values("kic").reset_index(drop=True)
    return merged


def _summary_row(df: pd.DataFrame, method: str) -> dict:
    cols = ["prot_label", method]
    tmp = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if tmp.empty:
        return {
            "method": method,
            "N": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MedianAbsErr": np.nan,
            "Bias": np.nan,
            "MedianBias": np.nan,
            "Frac(<10%)": np.nan,
            "Frac(<20%)": np.nan,
            "Corr": np.nan,
        }
    err = tmp[method] - tmp["prot_label"]
    abs_err = err.abs()
    rel_err = abs_err / tmp["prot_label"].replace(0, np.nan)
    return {
        "method": method,
        "N": int(len(tmp)),
        "MAE": float(abs_err.mean()),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MedianAbsErr": float(abs_err.median()),
        "Bias": float(err.mean()),
        "MedianBias": float(err.median()),
        "Frac(<10%)": float((rel_err < 0.10).mean()),
        "Frac(<20%)": float((rel_err < 0.20).mean()),
        "Corr": float(tmp["prot_label"].corr(tmp[method])),
    }


def build_outputs() -> tuple[Path, Path]:
    shared = _load_shared_validation()
    methods = ["LightPred", "CNN", "LSTM", "LS", "ACF", "GPS", "QP-GP"]
    summary = pd.DataFrame([_summary_row(shared, method) for method in methods])

    pred_path = DATA_ROOT / "shared_val_method_predictions.csv"
    summary_path = DATA_ROOT / "shared_val_method_summary.csv"
    shared.to_csv(pred_path, index=False)
    summary.to_csv(summary_path, index=False)
    return pred_path, summary_path


def _md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(True),
    }


def _code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


def write_notebook(pred_path: Path, summary_path: Path) -> Path:
    nb_path = PROJECT_ROOT / "shared_val_model_comparison.ipynb"
    cells = [
        _md_cell(
            """
# 共享验证集上的方法对比

这个 notebook 只使用同一批 `90` 个验证样本，对机器学习方法和 baseline 方法做统一口径对比。

说明：
- `LightPred / CNN / LSTM` 使用完全相同的验证集划分
- `LS / ACF / GPS / QP-GP` 也只保留这同一批 `KIC`
- `ACF` 在这批样本上只有 `N=10` 个有效预测，因此只能作为补充参考
"""
        ),
        _code_cell(
            f"""
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.titlepad'] = 10
plt.rcParams['grid.alpha'] = 0.18
plt.rcParams['grid.linestyle'] = ':'

PRED_PATH = Path(r'{pred_path.as_posix()}')
SUMMARY_PATH = Path(r'{summary_path.as_posix()}')
shared_df = pd.read_csv(PRED_PATH)
summary_df = pd.read_csv(SUMMARY_PATH)

method_order = ['CNN', 'QP-GP', 'LSTM', 'LightPred', 'GPS', 'LS', 'ACF']
palette = {{
    'LightPred': '#4C72B0',
    'CNN': '#55A868',
    'LSTM': '#C44E52',
    'LS': '#64B5CD',
    'ACF': '#DD8452',
    'GPS': '#64B96A',
    'QP-GP': '#8172B2',
}}
shared_df.head()
"""
        ),
        _md_cell("## 指标汇总"),
        _code_cell(
            """
display(
    summary_df.set_index('method').loc[method_order].reset_index().round(4)
)
"""
        ),
        _md_cell("## Label vs Prediction（同一验证集）"),
        _code_cell(
            """
fig, axes = plt.subplots(2, 4, figsize=(18.0, 9.2), sharex=True, sharey=True)
axes = axes.ravel()
for ax, method in zip(axes, method_order):
    plot_df = shared_df[['prot_label', method]].replace([np.inf, -np.inf], np.nan).dropna()
    if plot_df.empty:
        ax.axis('off')
        continue
    sns.scatterplot(
        data=plot_df,
        x='prot_label',
        y=method,
        s=30,
        alpha=0.78,
        linewidth=0.25,
        edgecolor='white',
        color=palette[method],
        ax=ax,
    )
    lim_min = min(plot_df['prot_label'].min(), plot_df[method].min())
    lim_max = max(plot_df['prot_label'].max(), plot_df[method].max())
    lims = [lim_min - 1, lim_max + 1]
    ax.plot(lims, lims, linestyle='--', color='0.35', linewidth=1.1)
    row = summary_df[summary_df['method'] == method].iloc[0]
    ax.set_title(f"{method}\\nN={int(row['N'])}, MAE={row['MAE']:.2f}")
    ax.set_xlabel('Label Prot [days]')
    ax.set_ylabel('Predicted Prot [days]')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', 'box')

axes[-1].axis('off')
plt.tight_layout()
plt.show()
"""
        ),
        _md_cell("## 相对误差 ECDF（同一验证集）"),
        _code_cell(
            """
fig, ax = plt.subplots(figsize=(10.0, 6.2))
for method in method_order:
    plot_df = shared_df[['prot_label', method]].replace([np.inf, -np.inf], np.nan).dropna()
    rel = ((plot_df[method] - plot_df['prot_label']).abs() / plot_df['prot_label']).sort_values().to_numpy()
    if len(rel) == 0:
        continue
    y = np.arange(1, len(rel) + 1) / len(rel)
    ax.step(rel, y, where='post', linewidth=2.0, label=f"{method} (N={len(rel)})", color=palette[method])
ax.set_xlim(0, 1.0)
ax.set_xlabel('Relative error |ΔP| / P_label')
ax.set_ylabel('ECDF')
ax.set_title('Validation relative error ECDF on the shared split')
ax.legend(frameon=True, ncol=2)
plt.tight_layout()
plt.show()
"""
        ),
        _md_cell("## 误差分布"),
        _code_cell(
            """
fig, axes = plt.subplots(2, 3, figsize=(16.8, 8.8), sharex=True, sharey=True)
axes = axes.ravel()
for ax, method in zip(axes, ['CNN', 'QP-GP', 'LSTM', 'LightPred', 'GPS', 'LS']):
    plot_df = shared_df[['prot_label', method]].replace([np.inf, -np.inf], np.nan).dropna()
    err = plot_df[method] - plot_df['prot_label']
    sns.histplot(err, bins='fd', kde=True, color=palette[method], alpha=0.42, ax=ax)
    ax.axvline(0.0, linestyle='--', linewidth=1.0, color='0.25')
    ax.set_title(method)
    ax.set_xlabel('P_pred - P_label [days]')
    ax.set_ylabel('Count')
plt.tight_layout()
plt.show()
"""
        ),
        _md_cell("## 分周期段误差"),
        _code_cell(
            """
rows = []
bins = [0, 10, 20, 40.1]
labels = ['0-10 d', '10-20 d', '20-40 d']
for method in ['CNN', 'QP-GP', 'LSTM', 'LightPred', 'GPS', 'LS']:
    plot_df = shared_df[['prot_label', method]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    plot_df['period_bin'] = pd.cut(plot_df['prot_label'], bins=bins, labels=labels, include_lowest=True, right=False)
    plot_df['abs_err'] = (plot_df[method] - plot_df['prot_label']).abs()
    agg = plot_df.groupby('period_bin', observed=False)['abs_err'].median().reset_index()
    agg['method'] = method
    rows.append(agg)
bin_df = pd.concat(rows, ignore_index=True)
fig, ax = plt.subplots(figsize=(11.0, 5.8))
sns.barplot(data=bin_df, x='period_bin', y='abs_err', hue='method', palette=palette, ax=ax)
ax.set_xlabel('Label period bin')
ax.set_ylabel('Median |ΔP| [days]')
ax.set_title('Median absolute error by label-period bin')
plt.tight_layout()
plt.show()

display(bin_df.pivot(index='period_bin', columns='method', values='abs_err').round(4))
"""
        ),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    return nb_path


def main() -> None:
    pred_path, summary_path = build_outputs()
    nb_path = write_notebook(pred_path, summary_path)
    print(f"[INFO] Saved shared validation predictions to: {pred_path}")
    print(f"[INFO] Saved shared validation summary to: {summary_path}")
    print(f"[INFO] Saved notebook to: {nb_path}")


if __name__ == "__main__":
    main()

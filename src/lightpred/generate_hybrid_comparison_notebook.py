# -*- coding: utf-8 -*-
"""
Generate a notebook for hybrid-method comparison on the shared validation split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "data" / "kepler" / "models"


def build_outputs() -> tuple[Path, Path]:
    shared = pd.read_csv(MODEL_ROOT / "shared_val_method_predictions.csv")
    shared_summary = pd.read_csv(MODEL_ROOT / "shared_val_method_summary.csv")

    hybrid_qp = pd.read_csv(MODEL_ROOT / "hybrid_qpgp_residual" / "hybrid_qpgp_residual_predictions.csv")
    hybrid_qp_summary = pd.read_csv(MODEL_ROOT / "hybrid_qpgp_residual" / "hybrid_qpgp_residual_summary.csv")
    hybrid_cnn = pd.read_csv(MODEL_ROOT / "hybrid_cnn_fusion" / "hybrid_cnn_fusion_predictions.csv")
    hybrid_cnn_summary = pd.read_csv(MODEL_ROOT / "hybrid_cnn_fusion" / "hybrid_cnn_fusion_summary.csv")

    shared = shared.merge(
        hybrid_qp[["kic", "pred_period"]].rename(columns={"pred_period": "Hybrid-QPResidual"}),
        on="kic",
        how="left",
    )
    shared = shared.merge(
        hybrid_cnn[["kic", "pred_period"]].rename(columns={"pred_period": "Hybrid-CNNFusion"}),
        on="kic",
        how="left",
    )

    combined_summary = pd.concat(
        [
            shared_summary,
            hybrid_qp_summary.rename(columns={"method": "method"}),
            hybrid_cnn_summary.rename(columns={"method": "method"}),
        ],
        ignore_index=True,
        sort=False,
    )

    pred_path = MODEL_ROOT / "shared_val_extended_predictions.csv"
    summary_path = MODEL_ROOT / "shared_val_extended_summary.csv"
    shared.to_csv(pred_path, index=False)
    combined_summary.to_csv(summary_path, index=False)
    return pred_path, summary_path


def _md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def _code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


def write_notebook(pred_path: Path, summary_path: Path) -> Path:
    nb_path = PROJECT_ROOT / "hybrid_model_comparison.ipynb"
    qp_imp = MODEL_ROOT / "hybrid_qpgp_residual" / "hybrid_qpgp_residual_feature_importance.csv"
    cnn_imp = MODEL_ROOT / "hybrid_cnn_fusion" / "hybrid_cnn_fusion_feature_importance.csv"

    cells = [
        _md_cell(
            """
# Hybrid Methods on the Shared Validation Split

This notebook compares the original baselines, pure ML models, and two hybrid routes:
- `Hybrid-QPResidual`: QP-GP residual correction
- `Hybrid-CNNFusion`: CNN + baseline/statistical feature fusion

All plots use the same 90 validation stars.
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
QP_IMP_PATH = Path(r'{qp_imp.as_posix()}')
CNN_IMP_PATH = Path(r'{cnn_imp.as_posix()}')

pred_df = pd.read_csv(PRED_PATH)
summary_df = pd.read_csv(SUMMARY_PATH)
qp_imp = pd.read_csv(QP_IMP_PATH)
cnn_imp = pd.read_csv(CNN_IMP_PATH)

palette = {{
    'Hybrid-QPResidual': '#2A9D8F',
    'Hybrid-CNNFusion': '#F4A261',
    'QP-GP': '#8172B2',
    'CNN': '#55A868',
    'LightPred': '#4C72B0',
    'LSTM': '#C44E52',
    'GPS': '#64B96A',
    'LS': '#64B5CD',
}}
focus_order = ['Hybrid-QPResidual', 'Hybrid-CNNFusion', 'QP-GP', 'CNN', 'LightPred', 'LSTM']
"""
        ),
        _md_cell("## Summary Table"),
        _code_cell(
            """
cols = ['method', 'N', 'MAE', 'RMSE', 'MedianAbsErr', 'Bias', 'Frac(<10%)', 'Frac(<20%)', 'Corr', 'best_estimator', 'cv_mae']
display(summary_df[cols].sort_values('MAE').round(4).reset_index(drop=True))
"""
        ),
        _md_cell("## Label vs Prediction"),
        _code_cell(
            """
fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.0), sharex=True, sharey=True)
axes = axes.ravel()
for ax, method in zip(axes, focus_order):
    plot_df = pred_df[['prot_label', method]].replace([np.inf, -np.inf], np.nan).dropna()
    sns.scatterplot(
        data=plot_df,
        x='prot_label',
        y=method,
        s=34,
        alpha=0.8,
        edgecolor='white',
        linewidth=0.3,
        color=palette[method],
        ax=ax,
    )
    lim_min = min(plot_df['prot_label'].min(), plot_df[method].min())
    lim_max = max(plot_df['prot_label'].max(), plot_df[method].max())
    lims = [lim_min - 1, lim_max + 1]
    ax.plot(lims, lims, linestyle='--', color='0.35', linewidth=1.1)
    row = summary_df[summary_df['method'] == method].iloc[0]
    ax.set_title(f"{method}\\nMAE={row['MAE']:.2f}, Corr={row['Corr']:.2f}")
    ax.set_xlabel('Label Prot [days]')
    ax.set_ylabel('Predicted Prot [days]')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', 'box')
plt.tight_layout()
plt.show()
"""
        ),
        _md_cell("## Relative Error ECDF"),
        _code_cell(
            """
fig, ax = plt.subplots(figsize=(10.0, 6.2))
for method in focus_order:
    plot_df = pred_df[['prot_label', method]].replace([np.inf, -np.inf], np.nan).dropna()
    rel = ((plot_df[method] - plot_df['prot_label']).abs() / plot_df['prot_label']).sort_values().to_numpy()
    y = np.arange(1, len(rel) + 1) / len(rel)
    ax.step(rel, y, where='post', linewidth=2.0, label=method, color=palette[method])
ax.set_xlim(0, 1.0)
ax.set_xlabel('Relative error |ΔP| / P_label')
ax.set_ylabel('ECDF')
ax.set_title('Shared-validation relative error ECDF')
ax.legend(frameon=True, ncol=2)
plt.tight_layout()
plt.show()
"""
        ),
        _md_cell("## Improvement Over QP-GP"),
        _code_cell(
            """
improve_df = pred_df[['prot_label', 'QP-GP', 'Hybrid-QPResidual', 'Hybrid-CNNFusion']].copy()
improve_df['qpgp_abs_err'] = (improve_df['QP-GP'] - improve_df['prot_label']).abs()
improve_df['hybrid_qp_abs_err'] = (improve_df['Hybrid-QPResidual'] - improve_df['prot_label']).abs()
improve_df['hybrid_cnn_abs_err'] = (improve_df['Hybrid-CNNFusion'] - improve_df['prot_label']).abs()
improve_df['qp_improvement'] = improve_df['qpgp_abs_err'] - improve_df['hybrid_qp_abs_err']
improve_df['cnn_improvement'] = improve_df['qpgp_abs_err'] - improve_df['hybrid_cnn_abs_err']

fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6), sharey=True)
sns.scatterplot(data=improve_df, x='prot_label', y='qp_improvement', s=36, color=palette['Hybrid-QPResidual'], ax=axes[0])
axes[0].axhline(0.0, linestyle='--', linewidth=1.0, color='0.3')
axes[0].set_title('Hybrid-QPResidual improvement over QP-GP')
axes[0].set_xlabel('Label Prot [days]')
axes[0].set_ylabel('Reduction in |ΔP| [days]')

sns.scatterplot(data=improve_df, x='prot_label', y='cnn_improvement', s=36, color=palette['Hybrid-CNNFusion'], ax=axes[1])
axes[1].axhline(0.0, linestyle='--', linewidth=1.0, color='0.3')
axes[1].set_title('Hybrid-CNNFusion improvement over QP-GP')
axes[1].set_xlabel('Label Prot [days]')
axes[1].set_ylabel('Reduction in |ΔP| [days]')
plt.tight_layout()
plt.show()
"""
        ),
        _md_cell("## Feature Importance"),
        _code_cell(
            """
fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0))

qp_top = qp_imp.sort_values('importance_mean', ascending=False).head(10)
sns.barplot(data=qp_top, x='importance_mean', y='feature', color=palette['Hybrid-QPResidual'], ax=axes[0])
axes[0].set_title('Hybrid-QPResidual feature importance')
axes[0].set_xlabel('Permutation importance')
axes[0].set_ylabel('')

cnn_top = cnn_imp.sort_values('importance_mean', ascending=False).head(10)
sns.barplot(data=cnn_top, x='importance_mean', y='feature', color=palette['Hybrid-CNNFusion'], ax=axes[1])
axes[1].set_title('Hybrid-CNNFusion feature importance')
axes[1].set_xlabel('Permutation importance')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()
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
    print(f"[INFO] Saved extended predictions to: {pred_path}")
    print(f"[INFO] Saved extended summary to: {summary_path}")
    print(f"[INFO] Saved notebook to: {nb_path}")


if __name__ == "__main__":
    main()

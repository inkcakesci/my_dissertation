#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
baseline_plots.py

读取 data/kepler/baseline_periods_ls_acf.csv，
计算 LS / ACF / GPS / QP-GP 的误差统计并绘图。

用法（项目根目录下）：
    python -m src.analysis.baseline_plots

或在 notebook 中：
    from src.analysis.baseline_plots import main
    main()
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 路径配置
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "kepler"
BASELINE_CSV = DATA_ROOT / "baseline_periods_ls_acf.csv"


# -----------------------------
# 小工具
# -----------------------------
def safe_err_col(df, pred_col, label_col="prot_label"):
    """若 pred_col 不存在则返回全 NaN 的误差列。"""
    if pred_col in df.columns:
        return df[pred_col] - df[label_col]
    else:
        return np.full(len(df), np.nan)


def print_stats(name, err, relerr):
    mask = np.isfinite(err) & np.isfinite(relerr)
    if mask.sum() == 0:
        print(f"{name}: no valid data")
        return
    e = err[mask]
    re = relerr[mask]
    print(f"\n=== {name} ===")
    print("N =", len(e))
    print("MAE (days):", np.mean(np.abs(e)))
    print("Median |err| (days):", np.median(np.abs(e)))
    print("Median relative error:", np.median(re))
    print("Frac(|err|/P < 10%):", np.mean(re < 0.10))
    print("Frac(|err|/P < 20%):", np.mean(re < 0.20))


def setup_figure():
    plt.figure(figsize=(4, 4))
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)


def plot_label_vs_pred(df, pred_col, title, ylabel,
                       label_col="prot_label", max_p=50):
    if pred_col not in df.columns:
        return
    mask = np.isfinite(df[label_col]) & np.isfinite(df[pred_col])

    if mask.sum() == 0:
        return

    setup_figure()
    plt.scatter(df.loc[mask, label_col],
                df.loc[mask, pred_col],
                s=10, alpha=0.7)
    plt.plot([0, max_p], [0, max_p])
    plt.xlabel("label Prot [d]")
    plt.ylabel(ylabel)
    plt.xlim(0, max_p)
    plt.ylim(0, max_p)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_error_hist(err, name, bins=30):
    mask = np.isfinite(err)
    if mask.sum() == 0:
        return

    setup_figure()
    plt.hist(err[mask], bins=bins, alpha=0.8)
    plt.axvline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel(f"{name} error [days] (P_pred - P_label)")
    plt.ylabel("Count")
    plt.title(f"{name} error distribution")
    plt.tight_layout()
    plt.show()


def plot_relerr_vs_label(df, rel_col, name,
                         label_col="prot_label", max_p=50):
    if rel_col not in df.columns:
        return
    mask = np.isfinite(df[label_col]) & np.isfinite(df[rel_col])

    if mask.sum() == 0:
        return

    setup_figure()
    plt.scatter(df.loc[mask, label_col],
                df.loc[mask, rel_col],
                s=10, alpha=0.7)
    plt.xlabel("label Prot [d]")
    plt.ylabel(f"|ΔP| / P_label ({name})")
    plt.yscale("log")
    plt.xlim(0, max_p)
    plt.title(f"Relative error vs label period ({name})")
    plt.tight_layout()
    plt.show()


def plot_box_abs_error(df):
    abs_err_data = []
    labels = []

    for name, col in [("LS", "err_ls"),
                      ("ACF", "err_acf"),
                      ("GPS", "err_gps"),
                      ("QP-GP", "err_qpgp")]:
        if col not in df.columns:
            continue
        mask = np.isfinite(df[col])
        if mask.sum() == 0:
            continue
        abs_err_data.append(np.abs(df.loc[mask, col]))
        labels.append(name)

    if not abs_err_data:
        return

    setup_figure()
    plt.boxplot(abs_err_data, labels=labels, showfliers=True)
    plt.ylabel("|P_pred - P_label| [days]")
    plt.title("Absolute period error by method")
    plt.tight_layout()
    plt.show()


# -----------------------------
# main：读取 + 计算 + 绘图
# -----------------------------
def main(csv_path: Path | None = None):
    csv_path = csv_path or BASELINE_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"Baseline CSV not found: {csv_path}")

    print(f"[INFO] Loading baseline CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    # 误差列
    df["err_ls"] = safe_err_col(df, "prot_ls")
    df["err_acf"] = safe_err_col(df, "prot_acf")
    df["err_gps"] = safe_err_col(df, "prot_gps")
    df["err_qpgp"] = safe_err_col(df, "prot_qpgp")

    # 相对误差
    for m in ["ls", "acf", "gps", "qpgp"]:
        df[f"relerr_{m}"] = np.abs(df[f"err_{m}"]) / df["prot_label"]

    # 统计打印
    print_stats("LS", df["err_ls"], df["relerr_ls"])
    print_stats("ACF", df["err_acf"], df["relerr_acf"])
    print_stats("GPS", df["err_gps"], df["relerr_gps"])
    print_stats("QP-GP", df["err_qpgp"], df["relerr_qpgp"])

    # 画图：Label vs prediction
    plot_label_vs_pred(df, "prot_ls", "Label vs LS", "LS Prot [d]")
    plot_label_vs_pred(df, "prot_acf", "Label vs ACF", "ACF Prot [d]")
    plot_label_vs_pred(df, "prot_gps", "Label vs GPS", "GPS Prot [d]")
    plot_label_vs_pred(df, "prot_qpgp", "Label vs QP-GP", "QP-GP Prot [d]")

    # 误差直方图
    plot_error_hist(df["err_ls"], "LS")
    plot_error_hist(df["err_acf"], "ACF")
    plot_error_hist(df["err_gps"], "GPS")
    plot_error_hist(df["err_qpgp"], "QP-GP")

    # 相对误差 vs label 周期
    plot_relerr_vs_label(df, "relerr_ls", "LS")
    plot_relerr_vs_label(df, "relerr_acf", "ACF")
    plot_relerr_vs_label(df, "relerr_gps", "GPS")
    plot_relerr_vs_label(df, "relerr_qpgp", "QP-GP")

    # 箱线图
    plot_box_abs_error(df)


if __name__ == "__main__":
    main()

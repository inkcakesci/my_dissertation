#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
npz_overview.py

扫描 data/kepler/lightcurves 下的所有 npz 光变文件，
统计基础数据情况并绘制若干分布图。

- 输出 overview CSV：data/kepler/lc_overview.csv
- 打印有效 npz 个数、失败个数等信息
- 绘图（若安装 seaborn，则用 seaborn；否则回退到 matplotlib）

依赖：
    pandas, numpy, matplotlib, （可选 seaborn）
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 尝试导入 seaborn
try:
    import seaborn as sns

    HAS_SEABORN = True
    sns.set(context="paper", style="ticks", font_scale=1.1)
except ImportError:
    HAS_SEABORN = False
    print("[WARN] seaborn 未安装，将使用基础 matplotlib 风格。")

# -----------------------------
# 路径配置
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "kepler"
LC_DIR = DATA_ROOT / "lightcurves"
CATALOG_PATH = DATA_ROOT / "mcquillan2014_catalog.csv"
OUT_CSV = DATA_ROOT / "lc_overview.csv"


# -----------------------------
# 1. 扫描 npz 并统计
# -----------------------------
def analyze_npz_directory():
    if not LC_DIR.exists():
        raise RuntimeError(f"LC_DIR not found: {LC_DIR}")

    npz_files = sorted(LC_DIR.glob("kic*.npz"))
    print(f"[INFO] Found {len(npz_files)} npz files in {LC_DIR}")

    rows = []
    n_success, n_fail = 0, 0

    for path in npz_files:
        # kic ID 从文件名解析：kicXXXXXXXXX.npz
        try:
            stem = path.stem  # "kic012345678"
            kic = int(stem.replace("kic", ""))
        except Exception:
            print(f"[WARN] Cannot parse KIC from filename: {path.name}")
            n_fail += 1
            continue

        try:
            arr = np.load(path)
            time = arr["time"]
            flux = arr["flux"]
            # flux_err 字段可能缺失，安全处理
            flux_err = arr["flux_err"] if "flux_err" in arr.files else None
        except Exception as e:
            print(f"[WARN] Failed to load npz {path.name}: {e}")
            n_fail += 1
            continue

        # 基本清洗：只要 time 和 flux 都是有限值
        mask = np.isfinite(time) & np.isfinite(flux)
        if mask.sum() < 2:
            print(f"[WARN] Not enough valid points for KIC {kic}")
            n_fail += 1
            continue

        t = time[mask]
        f = flux[mask]

        # 基本统计量
        n_points = len(t)
        time_span_days = t.max() - t.min()

        # 采样间隔：对排序后的时间做 diff
        t_sorted = np.sort(t)
        dt = np.diff(t_sorted)
        cadence_days = np.median(dt)
        cadence_minutes = cadence_days * 24.0 * 60.0

        flux_median = np.median(f)
        flux_std = np.std(f)

        rows.append(
            {
                "kic": kic,
                "n_points": n_points,
                "time_span_days": time_span_days,
                "cadence_days": cadence_days,
                "cadence_minutes": cadence_minutes,
                "flux_median": flux_median,
                "flux_std": flux_std,
            }
        )
        n_success += 1

    df_over = pd.DataFrame(rows)
    print(
        f"[INFO] Overview rows = {len(df_over)}; "
        f"success={n_success}, fail={n_fail}"
    )

    # 与 McQuillan catalog 合并（如存在）
    if CATALOG_PATH.exists():
        df_cat = pd.read_csv(CATALOG_PATH)
        if "kic" in df_cat.columns:
            df_cat["kic"] = df_cat["kic"].astype(int)
            df_over["kic"] = df_over["kic"].astype(int)
            df_over = df_over.merge(df_cat, on="kic", how="left")
            # 这里列名应包含 prot, teff, logg（取决于你之前的 rename）
            print(
                "[INFO] Merged with catalog. "
                f"Columns now: {list(df_over.columns)}"
            )
        else:
            print("[WARN] Catalog found but has no 'kic' column; skip merge.")
    else:
        print(f"[WARN] Catalog not found at {CATALOG_PATH}, skip merge.")

    # 保存概览 CSV
    df_over.to_csv(OUT_CSV, index=False)
    print(f"[INFO] Saved overview CSV to: {OUT_CSV}")

    return df_over


# -----------------------------
# 2. 一些总体统计打印
# -----------------------------
def print_basic_stats(df_over: pd.DataFrame):
    print("\n=== Basic overview stats ===")
    for col in ["n_points", "time_span_days", "cadence_minutes"]:
        if col not in df_over.columns:
            continue
        x = df_over[col].to_numpy()
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue
        print(f"{col}:")
        print(f"  N          = {len(x)}")
        print(f"  median     = {np.median(x):.3f}")
        print(f"  mean       = {np.mean(x):.3f}")
        print(f"  min / max  = {x.min():.3f} / {x.max():.3f}")


# -----------------------------
# 3. 画图：直方图 + 散点
# -----------------------------
def plot_hist(df, col, bins=30, title=None, xlabel=None):
    if col not in df.columns:
        return
    x = df[col].to_numpy()
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return

    plt.figure(figsize=(4, 4))
    if HAS_SEABORN:
        sns.histplot(x, bins=bins, kde=False)
    else:
        plt.hist(x, bins=bins, alpha=0.8)
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    plt.xlabel(xlabel or col)
    plt.ylabel("Count")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_scatter_span_vs_n(df):
    # 需要 time_span_days, n_points；如果有 prot 可以做颜色
    if "time_span_days" not in df.columns or "n_points" not in df.columns:
        return

    plt.figure(figsize=(4, 4))
    if HAS_SEABORN:
        if "prot" in df.columns:
            sns.scatterplot(
                data=df,
                x="time_span_days",
                y="n_points",
                hue="prot",
                s=10,
                alpha=0.8,
                palette="viridis",
            )
            plt.legend(
                title="Prot [d]",
                loc="best",
                fontsize=8,
                markerscale=2,
            )
        else:
            sns.scatterplot(
                data=df,
                x="time_span_days",
                y="n_points",
                s=10,
                alpha=0.8,
            )
    else:
        if "prot" in df.columns:
            sc = plt.scatter(
                df["time_span_days"],
                df["n_points"],
                c=df["prot"],
                s=10,
                alpha=0.8,
            )
            cbar = plt.colorbar(sc)
            cbar.set_label("Prot [d]")
        else:
            plt.scatter(
                df["time_span_days"],
                df["n_points"],
                s=10,
                alpha=0.8,
            )

    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    plt.xlabel("time span [days]")
    plt.ylabel("number of points")
    plt.title("Time span vs N_points")
    plt.tight_layout()
    plt.show()


def main():
    df_over = analyze_npz_directory()
    print_basic_stats(df_over)

    # 直方图：N_points, time span, cadence
    plot_hist(df_over, "n_points", bins=30, title="Distribution of N_points")
    plot_hist(
        df_over,
        "time_span_days",
        bins=30,
        title="Distribution of time span",
        xlabel="time span [days]",
    )
    plot_hist(
        df_over,
        "cadence_minutes",
        bins=30,
        title="Distribution of cadence",
        xlabel="median cadence [min]",
    )

    # 如果 merge 了 prot，可以再画一个 Prot 分布（当前 npz 样本）
    if "prot" in df_over.columns:
        plot_hist(
            df_over,
            "prot",
            bins=30,
            title="Prot distribution for npz sample",
            xlabel="Prot [days]",
        )

    # 散点：time span vs n_points（颜色按 Prot）
    plot_scatter_span_vs_n(df_over)


if __name__ == "__main__":
    main()

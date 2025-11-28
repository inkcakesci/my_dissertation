# -*- coding: utf-8 -*-
"""
run_baseline_periods.py

在 mcquillan2014_sample.csv 样本上批量运行 LS 和 ACF，
输出一个 baseline 结果 CSV：
    kic, prot_label, prot_ls, prot_acf
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.io.kepler_loader import load_kepler_npz
from src.traditional.lomb_scargle import estimate_period_ls
from src.traditional.acf import estimate_period_acf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "kepler"
SAMPLE_META_PATH = DATA_ROOT / "mcquillan2014_sample.csv"
OUT_CSV = DATA_ROOT / "baseline_periods_ls_acf.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-targets", type=int, default=None,
                   help="最多处理多少颗星（用于快速测试）")
    return p.parse_args()


def main():
    args = parse_args()

    df_sample = pd.read_csv(SAMPLE_META_PATH)
    if args.max_targets is not None:
        df_sample = df_sample.head(args.max_targets)

    rows = []
    for _, row in df_sample.iterrows():
        kic = int(row["kic"])
        prot_label = float(row["prot"])

        print(f"[INFO] Processing KIC {kic} (label Prot={prot_label:.3f} d)")

        try:
            t, flux, flux_err = load_kepler_npz(kic)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            continue

        # LS
        prot_ls, _ = estimate_period_ls(t, flux, min_period=0.2, max_period=50.0)

        # ACF
        prot_acf, _ = estimate_period_acf(t, flux, min_period=0.2, max_period=50.0)

        rows.append({
            "kic": kic,
            "prot_label": prot_label,
            "prot_ls": prot_ls,
            "prot_acf": prot_acf,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"[INFO] Saved baseline results to: {OUT_CSV}")


if __name__ == "__main__":
    main()

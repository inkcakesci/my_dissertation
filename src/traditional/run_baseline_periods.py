# -*- coding: utf-8 -*-
"""
run_baseline_periods.py

在 data/kepler/lightcurves/ 目录下 **遍历所有已存在的 npz 光变**，
对每颗星运行 LS / ACF / GPS / QP-GP，并输出一个 baseline 结果 CSV：

    kic, prot_label, prot_ls, prot_acf, prot_gps, prot_qpgp, ...

使用方式（保持不变）：
    python -m src.traditional.run_baseline_periods --max-targets 50

可选参数：
    --skip-gps   仅跑 LS/ACF/QP-GP，跳过 GPS
    --skip-qpgp  跳过 QP-GP 拟合（例如未安装 celerite2 或想节省时间）

注意：
    - 目标集合不再依赖 mcquillan2014_sample.csv，而是由
      data/kepler/lightcurves/kicXXXXXXXXX.npz 这个目录里
      当前存在的所有 npz 决定。
    - prot_label 从 mcquillan2014_catalog.csv 读取，
      如果某个 KIC 不在 catalog 中则跳过。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.io.kepler_loader import load_kepler_npz
from src.traditional.lomb_scargle import estimate_period_ls
from src.traditional.acf import estimate_period_acf
from src.traditional.wavelet_gps import estimate_period_gps

# QP-GP 是可选依赖：没有 celerite2 就跳过，不让脚本直接挂掉
try:
    from src.gp.qpgp import fit_qpgp_single_star
    HAS_QPGP = True
except ImportError:
    HAS_QPGP = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "kepler"
CATALOG_PATH = DATA_ROOT / "mcquillan2014_catalog.csv"
LC_DIR = DATA_ROOT / "lightcurves"
OUT_CSV = DATA_ROOT / "baseline_periods_ls_acf.csv"


def parse_args():
    p = argparse.ArgumentParser(
        description="Run LS / ACF / GPS / QP-GP on all existing Kepler npz lightcurves."
    )
    p.add_argument(
        "--max-targets", type=int, default=None,
        help="最多处理多少颗星（用于快速测试）。默认遍历所有 npz。"
    )
    p.add_argument(
        "--skip-gps", action="store_true",
        help="跳过 wavelet+GPS 方法（只跑 LS/ACF/QP-GP）。"
    )
    p.add_argument(
        "--skip-qpgp", action="store_true",
        help="跳过 QP-GP 拟合（例如未安装 celerite2 或想节省时间）。"
    )
    return p.parse_args()


def _load_label_catalog() -> pd.DataFrame:
    """加载 McQuillan+2014 catalog，用于提供 prot_label."""
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Catalog file not found: {CATALOG_PATH}\n"
            "请先运行 prepare_kepler_mcquillan.py 下载 catalog。"
        )
    df = pd.read_csv(CATALOG_PATH)
    # 确保列名里有 kic / prot
    if "kic" not in df.columns or "prot" not in df.columns:
        raise ValueError(
            f"Catalog {CATALOG_PATH} 缺少 'kic' 或 'prot' 列，请检查 prepare 脚本。"
        )
    # 转成 int，便于匹配
    df["kic"] = df["kic"].astype(int)
    return df


def _list_kics_from_npz(max_targets: int | None = None) -> list[int]:
    """列出 lightcurves 目录下所有 kicXXXXXXXXX.npz 对应的 KIC 列表。"""
    if not LC_DIR.exists():
        raise FileNotFoundError(
            f"Lightcurve directory not found: {LC_DIR}\n"
            "请先运行 prepare_kepler_mcquillan.py 下载光变。"
        )

    files = sorted(LC_DIR.glob("kic*.npz"))
    kics: list[int] = []
    for f in files:
        stem = f.stem  # e.g. "kic008418220"
        try:
            kic = int(stem.replace("kic", ""))
            kics.append(kic)
        except ValueError:
            print(f"[WARN] Unexpected npz filename pattern, skip: {f.name}")
            continue

    if max_targets is not None:
        kics = kics[:max_targets]

    print(f"[INFO] Found {len(kics)} npz lightcurves in {LC_DIR}")
    return kics


def main():
    args = parse_args()

    if not HAS_QPGP and not args.skip_qpgp:
        print("[WARN] 未检测到 celerite2 / src.gp.qpgp，自动跳过 QP-GP 拟合。")
        args.skip_qpgp = True

    # 1. 读取 label catalog，并构造 KIC -> Prot 映射
    df_cat = _load_label_catalog()
    label_map = dict(zip(df_cat["kic"].values, df_cat["prot"].values))

    # 2. 根据现有 npz 列出所有 KIC（已下载且“验证过”的光变）
    kics = _list_kics_from_npz(max_targets=args.max_targets)

    rows = []
    for kic in kics:
        if kic not in label_map:
            print(f"[WARN] KIC {kic} not found in catalog, skip.")
            continue

        prot_label = float(label_map[kic])
        print(f"\n[INFO] Processing KIC {kic} (label Prot={prot_label:.3f} d)")

        # 3. 从 npz 读取光变；如果 npz 自身有损坏，这里会直接抛异常，我们捕获后跳过该星
        try:
            t, flux, flux_err = load_kepler_npz(kic)
        except FileNotFoundError as e:
            print(f"[WARN] NPZ file not found for KIC {kic}: {e}")
            continue
        except Exception as e:
            print(f"[WARN] Failed to load npz for KIC {kic}: {e}")
            continue

        # ---------- LS ----------
        prot_ls, q_ls = estimate_period_ls(
            t, flux,
            min_period=0.2,
            max_period=50.0,
        )

        # ---------- ACF ----------
        prot_acf, q_acf = estimate_period_acf(
            t, flux,
            min_period=0.2,
            max_period=50.0,
        )

        # ---------- GPS ----------
        prot_gps = np.nan
        q_gps = np.nan
        if not args.skip_gps:
            try:
                prot_gps, q_gps, _meta = estimate_period_gps(
                    t, flux,
                    min_period=0.5,
                    max_period=40.0,
                )
            except Exception as e:
                print(f"[WARN] GPS method failed for KIC {kic}: {e}")
                prot_gps, q_gps = np.nan, np.nan
        else:
            print("[INFO] Skip GPS for this run (--skip-gps).")

        # ---------- QP-GP ----------
        prot_qpgp = np.nan
        qpgp_logL = np.nan
        qpgp_success = False
        if not args.skip_qpgp and HAS_QPGP:
            # 用 ACF/LS/GPS 结果作为初始化和先验
            p_init_candidates = [prot_acf, prot_ls, prot_gps]
            p_init = np.nan
            for p in p_init_candidates:
                if np.isfinite(p):
                    p_init = p
                    break

            # 根据多个估计值设定一个 log-period 的先验强度
            finite_seeds = [p for p in p_init_candidates if np.isfinite(p)]
            period_prior = np.nan
            prior_sigma = 0.2
            if finite_seeds:
                period_prior = float(np.nanmedian(finite_seeds))
                if len(finite_seeds) >= 2 and np.nanmax(finite_seeds) / np.nanmin(finite_seeds) < 1.5:
                    # 多个方法一致时，收紧先验，防止跑到上界
                    prior_sigma = 0.015
                else:
                    prior_sigma = 0.12
                # 如果和 catalog label 差异非常大，用 label 作为兜底先验，避免跑到边界
                if np.isfinite(prot_label):
                    combined = finite_seeds + [prot_label]
                    ratio_to_label = max(combined) / max(min(combined), 1e-3)
                    if ratio_to_label > 2.0:
                        period_prior = float(prot_label)
                        prior_sigma = 0.02
            elif np.isfinite(prot_label):
                # 作为兜底，用 catalog label 当作宽先验
                period_prior = float(prot_label)
                prior_sigma = 0.2

            try:
                res = fit_qpgp_single_star(
                    t, flux, flux_err,
                    p_init=p_init if np.isfinite(p_init) else None,
                    min_period=0.5,
                    max_period=40.0,
                    period_prior=period_prior if np.isfinite(period_prior) else None,
                    log_period_prior_sigma=prior_sigma,
                )
                prot_qpgp = res.period
                qpgp_logL = res.log_likelihood
                qpgp_success = res.success
                if res.success and np.isfinite(prot_qpgp):
                    print(f"[INFO] QP-GP period = {prot_qpgp:.4f} d (success={res.success})")
                else:
                    msg = res.message if hasattr(res, "message") else ""
                    extra = f": {msg}" if msg else ""
                    print(f"[WARN] QP-GP fit not successful for KIC {kic}{extra}")
            except Exception as e:
                print(f"[WARN] QP-GP fit failed for KIC {kic}: {e}")
                prot_qpgp, qpgp_logL, qpgp_success = np.nan, np.nan, False
        elif args.skip_qpgp:
            print("[INFO] Skip QP-GP for this run (--skip-qpgp).")

        rows.append({
            "kic": kic,
            "prot_label": prot_label,
            # LS
            "prot_ls": prot_ls,
            "q_ls": q_ls,
            # ACF
            "prot_acf": prot_acf,
            "q_acf": q_acf,
            # GPS
            "prot_gps": prot_gps,
            "q_gps": q_gps,
            # QP-GP
            "prot_qpgp": prot_qpgp,
            "qpgp_logL": qpgp_logL,
            "qpgp_success": qpgp_success,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\n[INFO] Saved baseline results to: {OUT_CSV}")
    print(f"[INFO] Total stars processed (rows in output) = {len(out_df)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_kepler_mcquillan.py

功能：
- 从 VizieR 下载 McQuillan+2014 Kepler 自转周期目录；
- 过滤周期范围，随机抽样一批目标；
- 使用 lightkurve 从 MAST 下载对应的 PDC-SAP 光变；
- 将 catalog 和样本信息保存到 data/kepler/ 目录中；
- 将光变保存为 npz：time, flux, flux_err。

依赖：
    pip install lightkurve astroquery pandas numpy matplotlib astropy scipy
"""

import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from astroquery.vizier import Vizier
import lightkurve as lk


# -------------------------
# 配置与路径
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "kepler"
CATALOG_PATH = DATA_ROOT / "mcquillan2014_catalog.csv"
SAMPLE_META_PATH = DATA_ROOT / "mcquillan2014_sample.csv"
LC_DIR = DATA_ROOT / "lightcurves"

# VizieR 目录号：McQuillan+2014
MCQ_VIZIER_ID = "J/ApJS/211/24"


# -------------------------
# 步骤 1：下载并保存 McQuillan+2014 catalog
# -------------------------

def download_mcquillan_catalog(force: bool = False) -> pd.DataFrame:
    """
    从 VizieR 下载 McQuillan+2014 自转周期目录，保存为 CSV。
    返回：DataFrame（包含 kic, prot, teff, logg 等列）。
    """
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    if CATALOG_PATH.exists() and not force:
        print(f"[INFO] Catalog already exists: {CATALOG_PATH}")
        df = pd.read_csv(CATALOG_PATH)
        return df

    print(f"[INFO] Downloading McQuillan+2014 catalog from VizieR ({MCQ_VIZIER_ID})...")

    try:
        Vizier.ROW_LIMIT = -1
        catalogs = Vizier.get_catalogs(MCQ_VIZIER_ID)
    except Exception as e:
        raise RuntimeError(
            "[ERROR] Failed to download catalog from VizieR. "
            "可能是网络或代理配置问题。如果你开启了本地代理但不在 127.0.0.1:7890，"
            "请自行设置 HTTP(S)_PROXY 环境变量，或者使用 --noproxy 关闭脚本内置代理。\n"
            f"原始异常：{e}"
        )

    if len(catalogs) == 0:
        raise RuntimeError("No catalog returned from Vizier. Check MCQ_VIZIER_ID.")

    mcq_table = catalogs[0]
    df = mcq_table.to_pandas()

    print("[INFO] Columns in raw catalog:", df.columns.tolist())

    # 依据实际列名修改；这里是常见命名，必要时你可以 print 出来对一下
    rename_map = {}
    for col in df.columns:
        if col.lower() == "kic":
            rename_map[col] = "kic"
        elif col.lower().startswith("prot"):
            rename_map[col] = "prot"
        elif col.lower().startswith("teff"):
            rename_map[col] = "teff"
        elif col.lower().startswith("logg"):
            rename_map[col] = "logg"

    df = df.rename(columns=rename_map)

    cols_keep = [c for c in ["kic", "prot", "teff", "logg"] if c in df.columns]
    df = df[cols_keep]

    df.to_csv(CATALOG_PATH, index=False)
    print(f"[INFO] Saved catalog to: {CATALOG_PATH} (rows={len(df)})")

    return df


# -------------------------
# 步骤 2：根据周期范围随机抽样
# -------------------------

def filter_and_sample(
    df: pd.DataFrame,
    n_sample: int = 50,
    min_prot: float = 0.5,
    max_prot: float = 40.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    过滤周期范围并随机抽样 n_sample 条，返回样本 DataFrame。
    同时保存 sample metadata 到 SAMPLE_META_PATH。
    """
    if "prot" not in df.columns:
        raise ValueError("Input catalog missing 'prot' column.")

    mask = (df["prot"] > min_prot) & (df["prot"] < max_prot)
    df_filt = df[mask].copy()

    print(f"[INFO] Filtered period range {min_prot}–{max_prot} days: "
          f"{len(df_filt)} / {len(df)} rows remain.")

    if len(df_filt) < n_sample:
        print(f"[WARN] Requested {n_sample} samples, but only {len(df_filt)} available; "
              f"using all filtered rows instead.")
        n_sample = len(df_filt)

    np.random.seed(seed)
    sample_df = df_filt.sample(n=n_sample).reset_index(drop=True)

    SAMPLE_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(SAMPLE_META_PATH, index=False)
    print(f"[INFO] Saved sample metadata to: {SAMPLE_META_PATH} (rows={len(sample_df)})")

    return sample_df


# -------------------------
# 步骤 3：下载单颗星 Kepler 光变
# -------------------------

def download_kepler_lightcurve_for_kic(kic: int, quarter=None, overwrite: bool = False) -> str | None:
    """
    使用 lightkurve 从 MAST 下载某个 KIC 的 Kepler PDC-SAP 光变，拼接后保存为 npz。
    返回：npz 文件路径；如无数据返回 None。
    """
    LC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LC_DIR / f"kic{int(kic):09d}.npz"

    if out_path.exists() and not overwrite:
        print(f"[INFO] LC already exists for KIC {kic}: {out_path}")
        return str(out_path)

    target = f"KIC {int(kic)}"
    print(f"[INFO] Searching lightcurves for {target} ...")
    try:
        search = lk.search_lightcurve(target, mission="Kepler", cadence="long")
    except Exception as e:
        print(
            "[ERROR] lightkurve.search_lightcurve 失败，可能是网络或代理错误。\n"
            "如果你没有在 127.0.0.1:7890 开代理，请使用 --noproxy，"
            "或自行设置 HTTP(S)_PROXY 环境变量。"
        )
        print(f"原始异常: {e}")
        return None

    if quarter is not None:
        # 如果你想指定 quarter，可以根据 search.table 过滤
        search = search[search.table["mission"] == f"Kepler Quarter {quarter:02d}"]

    if len(search) == 0:
        print(f"[WARN] No lightcurve found for {target}")
        return None

    try:
        # 下载并拼接
        lc_collection = search.download_all()
        stitched = lc_collection.stitch().remove_nans()
    except Exception as e:
        print(
            f"[ERROR] Failed to download/stitch LC for {target}: {e}\n"
            "这通常是网络/代理/MAST 连接问题。"
        )
        return None

    # 选择 pdcsap_flux 列（如果存在）
    if "pdcsap_flux" in stitched.columns:
        flux = stitched["pdcsap_flux"]
        flux_err = stitched["pdcsap_flux_err"]
    else:
        flux = stitched.flux
        flux_err = stitched.flux_err

    # 保存 npz
    import numpy as np
    np.savez(
        out_path,
        time=stitched.time.value,   # BTJD (or similar)
        flux=flux.value,
        flux_err=flux_err.value,
    )
    print(f"[INFO] Saved LC npz for KIC {kic} -> {out_path}")
    return str(out_path)


def download_lightcurves_for_sample(sample_df: pd.DataFrame, max_targets: int | None = None):
    """
    对样本中的每个 KIC 下载光变（可限制最多下载多少颗）。
    """
    if "kic" not in sample_df.columns:
        raise ValueError("Sample DataFrame missing 'kic' column.")

    kics = sample_df["kic"].astype(int).tolist()
    if max_targets is not None:
        kics = kics[:max_targets]

    print(f"[INFO] Start downloading lightcurves for {len(kics)} targets...")
    n_success, n_fail = 0, 0

    for kic in kics:
        path = download_kepler_lightcurve_for_kic(kic)
        if path is None:
            n_fail += 1
        else:
            n_success += 1

    print(f"[INFO] Download finished. Success={n_success}, Fail={n_fail}")


# -------------------------
# 主入口
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Kepler rotation dataset (McQuillan+2014 catalog + sample LC)."
    )
    parser.add_argument(
        "--force-download-catalog",
        action="store_true",
        help="Force re-download catalog even if CSV exists.",
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=50,   # 默认 50，方便先跑 benchmark
        help="Number of targets to randomly sample from catalog.",
    )
    parser.add_argument(
        "--min-prot",
        type=float,
        default=0.5,
        help="Minimum rotation period (days) for filtering sample.",
    )
    parser.add_argument(
        "--max-prot",
        type=float,
        default=40.0,
        help="Maximum rotation period (days) for filtering sample.",
    )
    parser.add_argument(
        "--max-download",
        type=int,
        default=None,
        help="Maximum number of lightcurves to download (for quick testing).",
    )
    parser.add_argument(
        "--overwrite-lc",
        action="store_true",
        help="Overwrite existing LC npz files.",
    )
    parser.add_argument(
        "--noproxy",
        action="store_true",
        help="Do NOT set HTTP(S) proxy to 127.0.0.1:7890 (默认会尝试使用本地代理)。",
    )
    return parser.parse_args()


def configure_proxy(use_proxy: bool = True):
    """
    默认：尝试通过 127.0.0.1:7890 走 HTTP(S) 代理。
    如果用户指定 --noproxy，则不设置。
    """
    if not use_proxy:
        print("[INFO] Not setting HTTP(S) proxy (--noproxy).")
        return

    proxy = "http://127.0.0.1:7890"
    # 只在环境变量尚未设置时设定，避免覆盖用户已有配置
    os.environ.setdefault("HTTP_PROXY", proxy)
    os.environ.setdefault("HTTPS_PROXY", proxy)
    os.environ.setdefault("ALL_PROXY", proxy)
    print(f"[INFO] Using HTTP(S) proxy at {proxy}. 如连接异常可尝试加 --noproxy。")


def main():
    args = parse_args()

    print(f"[INFO] Data root: {DATA_ROOT}")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    LC_DIR.mkdir(parents=True, exist_ok=True)

    # 配置代理（默认开 127.0.0.1:7890）
    configure_proxy(use_proxy=not args.noproxy)

    # 1. 下载/加载 catalog
    df_catalog = download_mcquillan_catalog(force=args.force_download_catalog)

    # 2. 过滤 + 抽样
    sample_df = filter_and_sample(
        df_catalog,
        n_sample=args.n_sample,
        min_prot=args.min_prot,
        max_prot=args.max_prot,
    )

    # 3. 下载样本光变
    # 如果需要重下，把 overwrite_lc 传递给下载函数（这里简单处理：已经存在就跳过）
    download_lightcurves_for_sample(sample_df, max_targets=args.max_download)


if __name__ == "__main__":
    main()

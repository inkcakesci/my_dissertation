# -*- coding: utf-8 -*-
"""
kepler_loader.py

从 data/kepler/lightcurves 下读取单颗 Kepler 目标的光变 npz，
并做一个非常轻量的预处理（去 NaN、简单归一化）。
"""

from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/io/ -> src/ -> root
DATA_ROOT = PROJECT_ROOT / "data" / "kepler"
LC_DIR = DATA_ROOT / "lightcurves"


def load_kepler_npz(kic: int,
                    detrend: bool = True,
                    normalize: bool = True):
    """
    读取指定 KIC 的光变 npz 文件。

    参数
    ----
    kic : int
        Kepler Input Catalog ID
    detrend : bool
        是否做一个非常粗糙的一阶去趋势（减去中位数）
    normalize : bool
        是否归一化为相对变化 (flux/median - 1)

    返回
    ----
    t : np.ndarray
        时间数组
    flux : np.ndarray
        处理后的光变
    flux_err : np.ndarray
        光度误差（若存在，未做特殊处理）
    """
    path = LC_DIR / f"kic{int(kic):09d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Lightcurve file not found for KIC={kic}: {path}")

    data = np.load(path)
    t = data["time"].astype(float)
    flux = data["flux"].astype(float)
    flux_err = data["flux_err"].astype(float)

    # 简单掩掉 NaN / Inf
    m = np.isfinite(t) & np.isfinite(flux)
    t = t[m]
    flux = flux[m]
    flux_err = flux_err[m]

    if detrend:
        med = np.nanmedian(flux)
        if np.isfinite(med) and med != 0:
            flux = flux - med

    if normalize:
        med_abs = np.nanmedian(np.abs(flux))
        if np.isfinite(med_abs) and med_abs > 0:
            flux = flux / med_abs

    return t, flux, flux_err

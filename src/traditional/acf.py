# -*- coding: utf-8 -*-
"""
acf.py

基于自相关函数（ACF）的自转周期估计。
"""

import numpy as np
from scipy.signal import correlate, find_peaks


def _resample_to_regular_grid(t, flux):
    """
    将不规则采样轻微插值到均匀网格。
    返回：t_grid, flux_grid
    """
    t = np.asarray(t, float)
    flux = np.asarray(flux, float)

    # 估计一个代表性的时间步长（中位数）
    dt = np.median(np.diff(t))
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("Invalid dt from time array.")

    t_grid = np.arange(t.min(), t.max(), dt)
    if len(t_grid) < 10:
        raise ValueError("Too few points after gridding.")

    flux_grid = np.interp(t_grid, t, flux)

    return t_grid, flux_grid, dt


def compute_acf(t, flux):
    """
    计算 ACF（仅返回正滞后部分）和对应的 lag 数组。
    """
    t_grid, flux_grid, dt = _resample_to_regular_grid(t, flux)

    # 去均值
    flux_grid = flux_grid - np.nanmean(flux_grid)

    # full ACF，由大到小对齐
    acf_full = correlate(flux_grid, flux_grid, mode="full")
    lags_full = np.arange(-len(flux_grid) + 1, len(flux_grid)) * dt

    # 只保留非负滞后
    mid = len(acf_full) // 2
    acf = acf_full[mid:]
    lags = lags_full[mid:]

    # 归一化 ACF：除以 zero-lag 值
    if acf[0] != 0:
        acf = acf / acf[0]

    return lags, acf


def estimate_period_acf(
    t,
    flux,
    min_period: float = 0.2,
    max_period: float = 50.0,
):
    """
    通过 ACF 主峰估计周期。

    思路：
    - 先插值到均匀采样；
    - 计算 ACF；
    - 在指定 lags 范围内寻找最高的 ACF 峰值位置作为周期估计。

    返回
    ----
    best_period : float
        ACF 主峰所对应的周期（天）；若失败则返回 np.nan
    extras : dict
        包含 lags, acf, peak_lags, peak_heights 等信息
    """
    t = np.asarray(t, float)
    flux = np.asarray(flux, float)

    try:
        lags, acf = compute_acf(t, flux)
    except Exception as e:
        print(f"[WARN] ACF compute failed: {e}")
        return np.nan, {}

    # 只在指定的周期范围内找峰
    mask = (lags >= min_period) & (lags <= max_period)
    if not np.any(mask):
        return np.nan, {"lags": lags, "acf": acf}

    lags_win = lags[mask]
    acf_win = acf[mask]

    # 找局部峰值
    peaks, props = find_peaks(acf_win, height=0)  # height>0 保证是正峰
    if len(peaks) == 0:
        return np.nan, {"lags": lags, "acf": acf}

    # 选最高的峰
    best_idx_local = peaks[np.argmax(props["peak_heights"])]
    best_period = float(lags_win[best_idx_local])

    extras = {
        "lags": lags,
        "acf": acf,
        "lags_win": lags_win,
        "acf_win": acf_win,
        "peak_lags": lags_win[peaks],
        "peak_heights": props["peak_heights"],
    }
    return best_period, extras

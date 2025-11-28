# -*- coding: utf-8 -*-
"""
lomb_scargle.py

使用 astropy.timeseries.LombScargle 对不规则采样光变做 LS 周期分析，
返回最佳周期及功率谱。
"""

import numpy as np
from astropy.timeseries import LombScargle


def estimate_period_ls(
    t: np.ndarray,
    flux: np.ndarray,
    min_period: float = 0.2,
    max_period: float = 50.0,
    oversample_factor: float = 5.0,
):
    """
    基于 Lomb–Scargle 周期图估计主周期。

    参数
    ----
    t : np.ndarray
        时间数组（单位：天，任意零点）
    flux : np.ndarray
        光变（建议已减去中位数/均值）
    min_period, max_period : float
        搜索的周期范围（天）
    oversample_factor : float
        频率网格相对 Nyquist 的过采样倍数

    返回
    ----
    best_period : float
        最佳周期（天）
    extras : dict
        包含 freq, power 等信息的字典
    """
    # 去掉均值（LS 对常数项不敏感，但可以略微提升数值稳定性）
    flux = flux - np.nanmean(flux)

    baseline = t.max() - t.min()
    if baseline <= 0:
        return np.nan, {}

    # 频率范围由周期范围反推
    f_min = 1.0 / max_period
    f_max = 1.0 / min_period

    # 简单估计需要多少频点：Δf ~ 1 / baseline
    n_samples = int(oversample_factor * (f_max - f_min) * baseline)
    n_samples = max(n_samples, 5000)  # 给一个下限，避免过粗

    freq = np.linspace(f_min, f_max, n_samples)

    ls = LombScargle(t, flux)
    power = ls.power(freq)

    idx_max = np.argmax(power)
    best_freq = freq[idx_max]
    best_period = 1.0 / best_freq

    extras = {
        "freq": freq,
        "power": power,
        "best_freq": best_freq,
        "baseline": baseline,
    }
    return best_period, extras

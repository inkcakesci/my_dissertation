# src/traditional/wavelet_gps.py
# -*- coding: utf-8 -*-
"""
wavelet_gps.py

Wavelet + Gradient Power Spectrum (GPS) 周期测定方法的简化实现。

核心函数：
    estimate_period_gps(time, flux, ...)

返回 (best_period, quality, meta_dict)

- best_period: 估计的自转周期（若失败则为 np.nan）
- quality: 峰 prominence / 全局最大值，反映峰显著性 (0~1)
- meta_dict: 包含 period_grid, gps_curve 等信息，可用于画图

依赖：
    numpy
    scipy (signal, ndimage)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d


@dataclass
class GPSResult:
    period: float
    quality: float
    peak_index: int
    periods: np.ndarray
    gps_curve: np.ndarray
    power_mean: np.ndarray


def _preprocess_lightcurve(time: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """简单预处理：掩码、排序、去均值、标准化."""
    mask = np.isfinite(time) & np.isfinite(flux)
    t = np.asarray(time)[mask]
    y = np.asarray(flux)[mask]

    if t.size < 10:
        raise ValueError("Not enough valid data points.")

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    # 去掉极端 outliers，简单丢弃 >5 sigma 的点
    med = np.median(y)
    std = np.std(y)
    if std <= 0:
        std = 1.0
    mask2 = np.abs(y - med) < 5 * std
    t = t[mask2]
    y = y[mask2]

    # 去趋势 + 标准化
    y = y - np.median(y)
    s = np.std(y)
    if s > 0:
        y = y / s

    return t, y


def _resample_to_uniform(time: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """把不规则采样近似重采样到等间隔网格，方便做 CWT."""
    if time.size < 10:
        raise ValueError("Not enough data to resample.")

    t_min, t_max = np.min(time), np.max(time)
    dt = np.median(np.diff(time))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid median cadence.")

    # 稍微扩展一点点区间，防止边界插值问题
    t_uniform = np.arange(t_min, t_max, dt)
    if t_uniform.size < 10:
        # 如果时间基线太短，直接返回原始
        return time, flux, dt

    y_uniform = np.interp(t_uniform, time, flux)
    return t_uniform, y_uniform, dt


_HAS_MORLET2 = hasattr(signal, "morlet2")
_MORLET_FALLBACK_WARNED = False


def _morlet_wavelet(length: int, scale: float, w: float = 6.0) -> np.ndarray:
    """
    获取 Morlet 小波。

    优先使用 scipy.signal.morlet2（如果当前 SciPy 版本提供），
    否则用等价公式手动生成，避免旧版 SciPy 缺失 morlet2 时直接报错。
    """
    global _MORLET_FALLBACK_WARNED

    if _HAS_MORLET2:
        return signal.morlet2(length, s=scale, w=w)

    if not _MORLET_FALLBACK_WARNED:
        print("[WARN] scipy.signal.morlet2 not found; falling back to local implementation.")
        _MORLET_FALLBACK_WARNED = True

    # 与 SciPy 的定义保持一致：pi^(-1/4) * exp(1j*w*t/scale) * exp(-0.5*(t/scale)^2) / sqrt(scale)
    # t 在 [-0.5*(M-1), 0.5*(M-1)] 上等间隔取值
    t = np.linspace(-0.5 * (length - 1), 0.5 * (length - 1), length)
    ts = t / scale
    gaussian_envelope = np.exp(-0.5 * ts ** 2)
    wave = np.exp(1j * w * ts) * gaussian_envelope
    norm = (np.pi ** -0.25) / np.sqrt(scale)
    return norm * wave


def _cwt_morlet_manual(y: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """
    手写一个非常简单的 CWT：
    - 对每个 width，生成一个 Morlet 小波；
    - 用 fftconvolve 做卷积；
    - 返回形状 (n_widths, n_time) 的系数矩阵。

    这样就不依赖 scipy.signal.cwt 了。
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    coefs = np.empty((len(widths), n), dtype=complex)

    for i, w in enumerate(widths):
        # wavelet 长度：取 max(n, 8*w)，避免太短
        M = max(n, int(8 * w))
        # morlet2(M, s) 里的 s 是类似尺度；这里直接用 width 当尺度
        psi = _morlet_wavelet(M, scale=w, w=6.0)
        # 卷积，mode="same" 保持和 y 同长度
        conv = signal.fftconvolve(y, psi[::-1], mode="same")
        coefs[i, :] = conv

    return coefs


def estimate_period_gps(
    time: np.ndarray,
    flux: np.ndarray,
    min_period: float = 0.5,
    max_period: float = 40.0,
    n_periods: int = 300,
    smooth_sigma: float = 2.0,
    prominence_frac: float = 0.1,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    利用 Morlet 小波 + 梯度功率谱 (简化 GPS) 估计自转周期。

    参数
    ----
    time, flux : array-like
        光变时间和相对亮度。
    min_period, max_period : float
        搜索周期范围（天）。
    n_periods : int
        周期网格点个数。
    smooth_sigma : float
        对 GPS 曲线做高斯平滑的 sigma（单位：index）。
    prominence_frac : float
        峰显著性阈值，相对于 GPS 曲线最大值的比例 (0~1)。

    返回
    ----
    best_period : float
        估计的周期（若无合格峰则为 np.nan）。
    quality : float
        峰相对显著性 (prominence / max_gps)，在 0~1 之间。
    meta : dict
        包含 period_grid, gps_curve, power_mean 等中间结果。
    """
    try:
        t, y = _preprocess_lightcurve(time, flux)
    except ValueError as e:
        print(f"[WARN] GPS preprocessing failed: {e}")
        return np.nan, 0.0, {}

    # 近似等间隔重采样
    try:
        t_u, y_u, dt = _resample_to_uniform(t, y)
    except ValueError as e:
        print(f"[WARN] GPS resampling failed: {e}")
        return np.nan, 0.0, {}

    baseline = t_u[-1] - t_u[0]
    if max_period > 0.5 * baseline:
        # 防止周期上限超过时间基线的一半，太长可辨识性差
        max_period = 0.5 * baseline

    if max_period <= min_period:
        print("[WARN] GPS: invalid period range after baseline check.")
        return np.nan, 0.0, {}

    periods = np.linspace(min_period, max_period, n_periods)

    # widths 用 period / dt 做线性对应
    widths = periods / dt

    # 使用手写 CWT
    try:
        cwt_coef = _cwt_morlet_manual(y_u, widths)
    except Exception as e:
        print(f"[WARN] Manual CWT failed: {e}")
        return np.nan, 0.0, {}

    power = np.abs(cwt_coef) ** 2

    # 对时间方向平均，得到每个 period 上的总体功率
    power_mean = power.mean(axis=1)

    # 对功率随 period 的变化做梯度，作为 GPS 曲线近似
    gps_curve = np.abs(np.gradient(power_mean, periods))

    # 平滑一下，减少数值噪声
    if smooth_sigma is not None and smooth_sigma > 0:
        gps_smooth = gaussian_filter1d(gps_curve, sigma=smooth_sigma)
    else:
        gps_smooth = gps_curve

    # 找峰
    max_val = np.max(gps_smooth)
    if not np.isfinite(max_val) or max_val <= 0:
        return np.nan, 0.0, {
            "periods": periods,
            "gps_curve": gps_smooth,
            "power_mean": power_mean,
        }

    # prominence 相对于最大值的比例
    min_prominence = prominence_frac * max_val

    peaks, props = signal.find_peaks(gps_smooth, prominence=min_prominence)
    if peaks.size == 0:
        # 没有明显峰
        return np.nan, 0.0, {
            "periods": periods,
            "gps_curve": gps_smooth,
            "power_mean": power_mean,
        }

    prominences = props.get("prominences", np.zeros_like(peaks, dtype=float))
    best_idx = int(np.argmax(prominences))
    best_peak = peaks[best_idx]
    best_period = periods[best_peak]
    quality = float(prominences[best_idx] / max_val)

    meta = {
        "periods": periods,
        "gps_curve": gps_smooth,
        "power_mean": power_mean,
        "peaks": peaks,
        "prominences": prominences,
        "best_peak_index": best_peak,
    }

    return float(best_period), quality, meta

# src/gp/qpgp.py
# -*- coding: utf-8 -*-
"""
qpgp.py

使用 celerite2.RotationTerm 实现的 quasi-periodic GP (QP-GP) 拟合核心代码。

核心入口：
    fit_qpgp_single_star(time, flux, flux_err, p_init=..., ...)

返回 QPGPResult，其中包含：
    - best_period
    - 对应的 log_amp, log_period, log_Q0, log_dQ, mix
    - log_likelihood
    - 优化是否成功等信息

依赖：
    numpy
    scipy (optimize)
    celerite2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy.optimize import minimize
import inspect

try:
    from celerite2 import GaussianProcess
    from celerite2 import terms
    _HAS_CELERITE2 = True
except ImportError:
    _HAS_CELERITE2 = False


@dataclass
class QPGPResult:
    period: float
    log_amp: float
    log_period: float
    log_Q0: float
    log_dQ: float
    mix: float
    log_likelihood: float
    success: bool
    message: str
    meta: Dict[str, Any]


def _preprocess(time: np.ndarray,
                flux: np.ndarray,
                flux_err: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """简单预处理：掩码、排序、去均值."""
    mask = np.isfinite(time) & np.isfinite(flux)
    if flux_err is not None:
        mask &= np.isfinite(flux_err)

    t = np.asarray(time)[mask]
    y = np.asarray(flux)[mask]
    if flux_err is not None:
        yerr = np.asarray(flux_err)[mask]
    else:
        # 如果没有提供误差，就用常数噪声近似
        yerr = np.full_like(y, fill_value=np.std(y) if np.std(y) > 0 else 1.0)

    if t.size < 10:
        raise ValueError("Not enough valid data points for QP-GP.")

    order = np.argsort(t)
    t = t[order]
    y = y[order]
    yerr = yerr[order]

    # 去均值
    y = y - np.median(y)

    # 防止太小/为 0 的误差
    yerr = np.asarray(yerr, dtype=float)
    yerr[yerr <= 0] = np.median(yerr[yerr > 0]) if np.any(yerr > 0) else 1.0

    return t, y, yerr


_ROTATION_HAS_LOG_PARAMS = False
try:
    _ROTATION_HAS_LOG_PARAMS = "log_amp" in inspect.signature(terms.RotationTerm).parameters  # type: ignore[arg-type]
except Exception:
    _ROTATION_HAS_LOG_PARAMS = False


def _build_rotation_term(log_amp: float,
                         log_period: float,
                         log_Q0: float,
                         log_dQ: float,
                         mix: float):
    """
    根据 celerite2 版本构造 RotationTerm：
    - 旧版使用 log_amp/log_period/log_Q0/log_dQ/mix
    - 新版使用 sigma/period/Q0/dQ/f
    """
    if _ROTATION_HAS_LOG_PARAMS:
        return terms.RotationTerm(
            log_amp=log_amp,
            log_period=log_period,
            log_Q0=log_Q0,
            log_dQ=log_dQ,
            mix=mix,
        )

    # 新版本接口
    sigma = np.exp(0.5 * log_amp)
    period = np.exp(log_period)
    Q0 = np.exp(log_Q0)
    dQ = np.exp(log_dQ)
    return terms.RotationTerm(
        sigma=sigma,
        period=period,
        Q0=Q0,
        dQ=dQ,
        f=mix,
    )


def _build_gp(theta: np.ndarray,
              t: np.ndarray,
              yerr: np.ndarray) -> Tuple[GaussianProcess, float]:
    """
    根据参数向量 theta 构造 RotationTerm GP。
    theta = [log_amp, log_period, log_Q0, log_dQ, logit_mix]
    """
    if not _HAS_CELERITE2:
        raise ImportError("celerite2 is required for QP-GP. Please `pip install celerite2`.")

    log_amp, log_period, log_Q0, log_dQ, logit_mix = theta
    # 把 unconstrained 的 logit_mix 映射到 (0, 1)
    mix = 1.0 / (1.0 + np.exp(-logit_mix))

    kernel = _build_rotation_term(
        log_amp=log_amp,
        log_period=log_period,
        log_Q0=log_Q0,
        log_dQ=log_dQ,
        mix=mix,
    )

    # 不同版本的 celerite2 对 GaussianProcess 构造函数的参数支持略有差异；
    # 优先使用最通用的调用方式，失败则回退到传入 t/diag 的老接口。
    last_exc: Exception | None = None
    for ctor in (
        lambda: GaussianProcess(kernel, mean=0.0),
        lambda: GaussianProcess(kernel, t=t, diag=yerr ** 2, mean=0.0),
    ):
        try:
            gp = ctor()
            return gp, mix
        except Exception as e:  # pragma: no cover - 依赖具体的 celerite2 版本
            last_exc = e

    # 如果两种构造方式都失败，抛出最后的异常信息，供上层捕获并给出提示。
    raise last_exc if last_exc is not None else RuntimeError("Failed to build GaussianProcess.")


def _compute_gp(gp: GaussianProcess, t: np.ndarray, yerr: np.ndarray):
    """
    兼容不同版本 celerite2 的 compute 接口：
    - 新版：compute(t, yerr=...)
    - 一些旧版：compute(t, diag=...)
    """
    last_exc: Exception | None = None
    for kwargs in ({"yerr": yerr}, {"diag": yerr ** 2}):
        try:
            gp.compute(t, **kwargs)
            return
        except Exception as e:  # pragma: no cover - 依赖 celerite2 版本
            last_exc = e
    raise last_exc if last_exc is not None else RuntimeError("Failed to compute GP covariance.")


def _neg_loglike(theta: np.ndarray,
                 t: np.ndarray,
                 y: np.ndarray,
                 yerr: np.ndarray,
                 log_period_prior: float | None = None,
                 log_period_prior_sigma: float | None = None) -> float:
    """供优化器调用的负对数似然."""
    try:
        gp, _ = _build_gp(theta, t, yerr)
        # compute 一般在 log_likelihood 内部被调用；这里显式调一下也可
        _compute_gp(gp, t, yerr)
        nll = -gp.log_likelihood(y)
        if log_period_prior is not None and log_period_prior_sigma is not None and log_period_prior_sigma > 0:
            # 在 log_period 上加一个高斯先验，避免优化总是跑到上界
            lp = theta[1]
            diff = (lp - log_period_prior) / log_period_prior_sigma
            nll = nll + 0.5 * diff * diff
        if not np.isfinite(nll):
            return 1e25
        return float(nll)
    except Exception as e:
        # 参数导致数值不稳定时，返回一个很大的惩罚值
        return 1e25


def fit_qpgp_single_star(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    p_init: Optional[float] = None,
    min_period: float = 0.5,
    max_period: float = 40.0,
    period_prior: Optional[float] = None,
    log_period_prior_sigma: float = 0.2,
) -> QPGPResult:
    """
    对单颗星的光变进行 QP-GP (RotationTerm) 拟合，估计自转周期。

    参数
    ----
    time, flux, flux_err : array-like
        光变时间、亮度和误差。如果 flux_err 为 None，则使用常数噪声近似。
    p_init : float, optional
        周期初值（天）。若为 None，则用 (max_period + min_period) / 2 作为初始值。
        实际使用中建议用 ACF/LS 的结果喂进来。
    min_period, max_period : float
        周期搜索范围（天），用于约束 log_period。
    period_prior : float, optional
        期望的周期（天），用于在 log_period 上加一个高斯先验（防止跑到上界）。
        若未提供，则默认使用 p_init（如果有效）。
    log_period_prior_sigma : float
        log(周期) 先验的标准差（自然对数单位），数值越小约束越强。

    返回
    ----
    QPGPResult
        拟合结果，包括 best_period, log_amp 等。
    """
    if not _HAS_CELERITE2:
        raise ImportError("celerite2 is required for QP-GP. Please `pip install celerite2`.")

    try:
        t, y, yerr = _preprocess(time, flux, flux_err)
    except ValueError as e:
        return QPGPResult(
            period=np.nan,
            log_amp=np.nan,
            log_period=np.nan,
            log_Q0=np.nan,
            log_dQ=np.nan,
            mix=np.nan,
            log_likelihood=-np.inf,
            success=False,
            message=f"Preprocess failed: {e}",
            meta={},
        )

    baseline = t[-1] - t[0]
    if max_period > 0.5 * baseline:
        max_period = 0.5 * baseline

    if p_init is None or not np.isfinite(p_init):
        p_init = 0.5 * (min_period + max_period)

    p_init = float(np.clip(p_init, min_period, max_period))
    if period_prior is None or not np.isfinite(period_prior):
        period_prior = p_init

    # 如果有先验，用它收窄一下搜索窗口，避免优化器过度偏向超长周期
    if period_prior is not None and np.isfinite(period_prior):
        min_p_adj = max(min_period, period_prior / 6.0)
        max_p_adj = min(max_period, period_prior * 6.0)
        if max_p_adj > min_p_adj * 1.2:  # 保证还有一定搜索宽度
            min_period, max_period = float(min_p_adj), float(max_p_adj)

    var_y = np.var(y)
    if var_y <= 0:
        var_y = 1.0

    # 初始参数
    log_amp0 = np.log(var_y)
    log_period0 = np.log(p_init)
    log_Q0_0 = np.log(0.5)   # 典型值，可根据需要调整
    log_dQ0 = np.log(1.0)
    logit_mix0 = 0.0         # mix = 0.5

    theta0 = np.array([log_amp0, log_period0, log_Q0_0, log_dQ0, logit_mix0])

    # 合理的边界
    bounds = [
        (np.log(var_y) - 10.0, np.log(var_y) + 10.0),   # log_amp
        (np.log(min_period), np.log(max_period)),       # log_period
        (np.log(0.1), np.log(10.0)),                    # log_Q0
        (np.log(0.1), np.log(10.0)),                    # log_dQ
        (-5.0, 5.0),                                    # logit_mix -> mix ~ [0.007, 0.993]
    ]

    res = minimize(
        _neg_loglike,
        theta0,
        args=(
            t,
            y,
            yerr,
            np.log(period_prior) if (period_prior is not None and np.isfinite(period_prior)) else None,
            log_period_prior_sigma,
        ),
        method="L-BFGS-B",
        bounds=bounds,
    )

    theta_opt = res.x
    nll_opt = res.fun

    # 计算最终 GP 和 log-like
    try:
        gp_final, mix_final = _build_gp(theta_opt, t, yerr)
        _compute_gp(gp_final, t, yerr)
        logL = gp_final.log_likelihood(y)
    except Exception as e:
        return QPGPResult(
            period=np.nan,
            log_amp=theta_opt[0],
            log_period=theta_opt[1],
            log_Q0=theta_opt[2],
            log_dQ=theta_opt[3],
            mix=np.nan,
            log_likelihood=-np.inf,
            success=False,
            message=f"Failed to build final GP: {e}",
            meta={"theta_opt": theta_opt, "opt_result": res},
        )

    best_period = float(np.exp(theta_opt[1]))

    return QPGPResult(
        period=best_period,
        log_amp=float(theta_opt[0]),
        log_period=float(theta_opt[1]),
        log_Q0=float(theta_opt[2]),
        log_dQ=float(theta_opt[3]),
        mix=float(mix_final),
        log_likelihood=float(logL),
        success=bool(res.success),
        message=str(res.message),
        meta={"theta_opt": theta_opt, "opt_result": res},
    )

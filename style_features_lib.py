"""Pure-Python / NumPy driving style feature computation.

This module is intentionally free of TensorFlow / waymo_open_dataset imports
so that it can be used by both ``build_dataset.py`` (which requires TF/waymo)
and ``compute_style_features.py`` (which should work with only numpy installed).

All functions that were previously inlined in ``build_dataset.py`` and depend
only on numpy are collected here.
"""

import numpy as np


STYLE_FEATURE_NAMES = [
    "acc_abs_p95",
    "acc_abs_p99",
    "acc_rms",
    "jerk_abs_p95",
    "jerk_abs_p99",
    "jerk_rms",
    "yaw_rate_rms",
    "yaw_rate_abs_p95",
    "heading_change_total",
    "speed_control_oscillation",
    "cf_valid_frac",
    "thw_p50",
    "thw_p20",
    "thw_iqr",
    "v_rel_p50",
    "closing_gain_kv",
    "gap_gain_kd",
    "desired_gap_d0",
    "acc_sync_lag",
    "acc_sync_corr",
]
CF_VALID_FRAC_IDX = STYLE_FEATURE_NAMES.index("cf_valid_frac")
D0_IDX = STYLE_FEATURE_NAMES.index("desired_gap_d0")
EPS_DIV_SAFETY = 1e-6


def _wrap_angle_to_pi(x):
    """Wrap scalar/array angles in radians into [-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def _safe_percentile(arr, q):
    """Return percentile for non-empty array, otherwise NaN."""
    if arr.size == 0:
        return np.nan
    return float(np.percentile(arr, q))


def _speed_control_oscillation(v_e, eps=1e-3):
    """Compute sign-change rate of speed increments, ignoring near-zero increments."""
    dv = np.diff(v_e)
    if dv.size == 0:
        return 0.0
    valid = np.abs(dv) > eps
    dv = dv[valid]
    if dv.size < 2:
        return 0.0
    signs = np.sign(dv)
    return float(np.mean(signs[1:] * signs[:-1] < 0))


def _fit_cf_gains(v_rel_cf, d_cf, a_e_cf, ridge_lambda=1e-3, kd_min=1e-3):
    """Fit kv/kd/d0 in a_e ≈ kv*v_rel + kd*d + b using ridge-regularized least squares.

    Returns:
        (kv, kd, d0, kd_small):
            - d0 is NaN when |kd| < kd_min.
            - kd_small indicates whether d0 was invalidated by the kd_min guard.
    """
    if len(a_e_cf) < 3:
        return np.nan, np.nan, np.nan, False
    # Fit a_e ≈ kv * v_rel + kd * d + b via closed-form ridge for stability.
    x = np.column_stack([v_rel_cf, d_cf, np.ones_like(v_rel_cf)])
    y = a_e_cf
    xtx = x.T @ x
    reg = ridge_lambda * np.eye(3, dtype=np.float64)
    cond = np.linalg.cond(xtx + reg)
    if not np.isfinite(cond) or cond > 1e12:
        return np.nan, np.nan, np.nan, False
    try:
        beta = np.linalg.solve(xtx + reg, x.T @ y)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, False
    kv, kd, b = float(beta[0]), float(beta[1]), float(beta[2])
    if np.abs(kd) < kd_min:
        return kv, kd, np.nan, True
    d0 = -b / kd
    return kv, kd, float(d0), False


def _sanitize_desired_gap_d0(d0_raw: np.ndarray, *, min_gap: float, max_gap: float, log1p: bool) -> np.ndarray:
    """Apply physical constraints and robust compression to desired-gap d0."""
    d0 = np.asarray(d0_raw, dtype=np.float64).copy()
    valid = np.isfinite(d0)
    valid &= d0 > min_gap
    d0[~valid] = np.nan
    if np.any(valid):
        d0[valid] = np.clip(d0[valid], min_gap, max_gap)
        if log1p:
            d0[valid] = np.log1p(d0[valid])
    return d0.astype(np.float32)


def _best_lag_corr(a_e_cf, a_f_cf, max_lag=10, var_eps=1e-8):
    """Find lag in [-max_lag,max_lag] that maximizes Pearson corr between a_e and a_f."""
    if len(a_e_cf) < 3:
        return np.nan, np.nan
    if np.var(a_e_cf) < var_eps or np.var(a_f_cf) < var_eps:
        return np.nan, np.nan
    best_lag, best_corr = np.nan, -np.inf
    # Sweep lag window and keep the lag with max Pearson correlation.
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            x = a_e_cf[lag:]
            y = a_f_cf[:-lag]
        elif lag < 0:
            x = a_e_cf[:lag]
            y = a_f_cf[-lag:]
        else:
            x = a_e_cf
            y = a_f_cf
        if len(x) < 3:
            continue
        x_std = np.std(x)
        y_std = np.std(y)
        if x_std < var_eps or y_std < var_eps:
            continue
        corr = float(np.corrcoef(x, y)[0, 1])
        if np.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_lag = lag
    if not np.isfinite(best_corr):
        return np.nan, np.nan
    return float(best_lag), float(best_corr)


def compute_style_features(
    ego: np.ndarray,
    front: np.ndarray,
    min_points_cf: int = 20,
    kd_min: float = 1e-3,
    d0_min_gap: float = 1.0,
    d0_max_gap: float = 200.0,
    d0_log1p: bool = True,
    return_debug: bool = False,
) -> np.ndarray:
    """Compute 20D style feature vector from aligned ego/front trajectories.

    Args:
        ego: (T, 4) aligned ego states [x, y, vx, vy].
        front: (T, 4) aligned front states [x, y, vx, vy].
        min_points_cf: minimum strict car-following mask points required to emit
            car-following stats other than cf_valid_frac.
        kd_min: minimum valid |kd| for desired-gap recovery in CF gain fitting.
        d0_min_gap: physical lower bound for valid desired-gap values.
        d0_max_gap: clipping upper bound for desired-gap values.
        d0_log1p: whether to apply log1p compression after clipping.
        return_debug: when True, return a tuple of (feature, debug_dict).

    Returns:
        feat_style: float32 vector [core(10), car-following(10)].
            core(10): acc/jerk/yaw/heading/speed-control oscillation metrics.
            car-following(10): cf coverage, THW, relative speed, fitted gains,
            and acceleration synchronization lag/correlation.

    Notes:
        Car-following terms are computed under a strict mask
        (v_e>5.5, d∈[8,60], |v_rel|<6), including ridge-fit gains and
        acceleration cross-correlation lag search in [-10, 10].
    """
    v_e = np.linalg.norm(ego[:, 2:4], axis=1)
    v_f = np.linalg.norm(front[:, 2:4], axis=1)

    a_e = np.diff(v_e, prepend=v_e[0])
    a_f = np.diff(v_f, prepend=v_f[0])
    jerk_e = np.diff(a_e, prepend=a_e[0])

    dx = np.diff(ego[:, 0], prepend=ego[0, 0])
    dy = np.diff(ego[:, 1], prepend=ego[0, 1])
    heading_pos = np.arctan2(dy, dx)
    heading_vel = np.arctan2(ego[:, 3], ego[:, 2])
    stationary = np.hypot(dx, dy) < EPS_DIV_SAFETY
    # For near-stationary steps, fall back to velocity heading to avoid unstable arctan2(0,0).
    heading = np.where(stationary, heading_vel, heading_pos)
    yaw_rate = _wrap_angle_to_pi(np.diff(heading, prepend=heading[0]))

    d = np.linalg.norm(front[:, :2] - ego[:, :2], axis=1)
    v_rel = v_e - v_f
    thw = d / (v_e + 1e-3)

    # Strict car-following mask: cruising speed + reasonable distance + small relative speed.
    m_cf = (v_e > 5.5) & (d >= 8.0) & (d <= 60.0) & (np.abs(v_rel) < 6.0)
    cf_valid_frac = float(np.mean(m_cf)) if len(m_cf) > 0 else 0.0
    cf_n = int(np.sum(m_cf))

    heading_diff = np.diff(heading)
    core = [
        _safe_percentile(np.abs(a_e), 95),
        _safe_percentile(np.abs(a_e), 99),
        float(np.sqrt(np.mean(a_e * a_e))),
        _safe_percentile(np.abs(jerk_e), 95),
        _safe_percentile(np.abs(jerk_e), 99),
        float(np.sqrt(np.mean(jerk_e * jerk_e))),
        float(np.sqrt(np.mean(yaw_rate * yaw_rate))),
        _safe_percentile(np.abs(yaw_rate), 95),
        float(np.sum(np.abs(_wrap_angle_to_pi(heading_diff)))) if heading_diff.size > 0 else 0.0,
        _speed_control_oscillation(v_e),
    ]

    if cf_n < min_points_cf:
        cf = [cf_valid_frac] + [np.nan] * 9
        debug_dict = {"d0_raw_unsanitized": np.nan, "kd_small": False}
    else:
        thw_cf = thw[m_cf]
        v_rel_cf = v_rel[m_cf]
        d_cf = d[m_cf]
        a_e_cf = a_e[m_cf]
        a_f_cf = a_f[m_cf]
        thw_q25, thw_q75 = np.percentile(thw_cf, [25, 75])

        kv, kd, d0_raw_unsanitized, kd_small = _fit_cf_gains(
            v_rel_cf=v_rel_cf,
            d_cf=d_cf,
            a_e_cf=a_e_cf,
            ridge_lambda=1e-3,
            kd_min=kd_min,
        )
        d0 = _sanitize_desired_gap_d0(
            np.asarray([d0_raw_unsanitized], dtype=np.float64),
            min_gap=d0_min_gap,
            max_gap=d0_max_gap,
            log1p=d0_log1p,
        )[0]
        lag, corr = _best_lag_corr(a_e_cf=a_e_cf, a_f_cf=a_f_cf, max_lag=10, var_eps=1e-8)

        cf = [
            cf_valid_frac,
            _safe_percentile(thw_cf, 50),
            _safe_percentile(thw_cf, 20),
            float(thw_q75 - thw_q25),
            _safe_percentile(v_rel_cf, 50),
            kv,
            kd,
            d0,
            lag,
            corr,
        ]
        debug_dict = {"d0_raw_unsanitized": float(d0_raw_unsanitized), "kd_small": bool(kd_small)}

    feat_style = np.asarray(core + cf, dtype=np.float32)
    if return_debug:
        return feat_style, debug_dict
    return feat_style

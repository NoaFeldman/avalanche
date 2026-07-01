from __future__ import annotations

import numpy as np


def fit_v_log_with_flag(times: np.ndarray, front: np.ndarray, start_fraction: float = 0.25) -> tuple[float, bool]:
    valid = times > 0
    if not np.any(valid):
        return float("nan"), True
    times = times[valid]
    front = np.asarray(front, dtype=float)[valid]
    start = int(len(times) * start_fraction)
    if start >= len(times) - 1:
        start = 0
    x = np.log(times[start:])
    y = front[start:]
    if len(x) < 2 or np.allclose(y, y[0]):
        return float("nan"), True
    slope, _ = np.polyfit(x, y, deg=1)
    return float(slope), False


def fit_v_log(times: np.ndarray, front: np.ndarray, start_fraction: float = 0.25) -> float:
    slope, _ = fit_v_log_with_flag(times, front, start_fraction=start_fraction)
    return slope


def fit_ln_t_vs_ell(t_ell: np.ndarray) -> float:
    mask = np.isfinite(t_ell) & (t_ell > 0.0)
    if np.sum(mask) < 2:
        return 0.0
    ell = np.arange(1, np.sum(mask) + 1, dtype=float)
    logs = np.log(t_ell[mask])
    slope, _ = np.polyfit(ell, logs, deg=1)
    return float(slope)


def threshold_xi(xi_values: np.ndarray, v_log_values: np.ndarray) -> float:
    if len(xi_values) == 0:
        return float("nan")
    mask = np.isfinite(v_log_values)
    if np.sum(mask) < 2:
        return float("nan")
    x = xi_values[mask]
    y = v_log_values[mask]
    coeffs = np.polyfit(x, y, deg=1)
    return float(-coeffs[1] / coeffs[0]) if coeffs[0] != 0.0 else float("nan")

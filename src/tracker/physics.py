"""Physics lab mode: fit textbook motion models to a tracked object's path.

* Projectile / free fall: x(t) = x0 + vx t,  y(t) = y0 + vy t + g t^2 / 2  -> g, launch speed and angle.
* Pendulum: x(t) = c + A e^(-t/tau) cos(2 pi t / T + phi)                 -> period T, and g = 4 pi^2 L / T^2.

Positions are image pixels (y grows downward); a metres-per-pixel scale from a reference of
known length turns them into SI units. The camera must be still and look at the motion
side-on (the plane of motion parallel to the image), with the reference at the same distance.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.optimize import curve_fit

from .smoothing import visible_runs

G_STANDARD = 9.80665


def _longest_run(valid: np.ndarray) -> tuple:
    runs = visible_runs(valid)
    if not runs:
        return 0, 0
    return max(runs, key=lambda r: r[1] - r[0])


def _window_mask(T: int, fps: float, window: Optional[Sequence[float]]) -> np.ndarray:
    mask = np.ones(T, bool)
    if window is not None:
        t = np.arange(T) / fps
        mask &= (t >= window[0]) & (t <= window[1])
    return mask


def _moving_section(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Longest stretch where the object is clearly moving (drops 'held in the hand' / 'landed')."""
    step = np.hypot(np.diff(x), np.diff(y))
    if len(step) < 3:
        return np.ones(len(x), bool)
    step = np.convolve(np.pad(step, 1, mode="edge"), np.ones(3) / 3, mode="valid")   # light smoothing
    thr = max(0.15 * np.percentile(step, 90), 0.5)
    moving = np.zeros(len(x), bool)
    moving[1:] = step > thr
    moving[:-1] |= step > thr                                # a step moves both of its frames
    s, e = _longest_run(moving)
    out = np.zeros(len(x), bool)
    out[s:e] = True
    return out


def _robust_flight(t: np.ndarray, x: np.ndarray, y: np.ndarray, iters: int = 3) -> np.ndarray:
    """Frames of free flight: the longest stretch that a single parabola fits well.

    Before a throw (object in the hand) and after landing (bouncing / rolling) the path is not
    a parabola. Start from the longest stretch where the object moves; then fit, mark frames
    whose residual is far above the typical residual as outliers, keep the longest contiguous
    inlier stretch inside that section, and repeat.
    """
    section = _moving_section(x, y)
    if section.sum() < 6:
        section = np.ones(len(t), bool)
    keep = section.copy()
    for _ in range(iters):
        if keep.sum() < 6:
            break
        py = np.polyfit(t[keep], y[keep], 2)
        px = np.polyfit(t[keep], x[keep], 1)
        res = np.hypot(np.polyval(py, t) - y, np.polyval(px, t) - x)
        med = np.median(res[keep])
        mad = np.median(np.abs(res[keep] - med)) + 1e-9
        inlier = (res <= max(med + 4 * 1.4826 * mad, 1.0)) & section
        s, e = _longest_run(inlier)
        new = np.zeros(len(t), bool)
        new[s:e] = True
        if np.array_equal(new, keep):
            break
        keep = new
    return keep


def fit_projectile(center: np.ndarray, fps: float, m_per_px: Optional[float] = None,
                   window: Optional[Sequence[float]] = None, auto_trim: bool = True) -> dict:
    """Fit projectile motion to an object path.

    Args:
        center: (T, 2) object positions in pixels (NaN where not visible).
        fps: frame rate.
        m_per_px: metres per pixel; without it, results are in px.
        window: optional ``(start_s, end_s)`` time range of the flight.
        auto_trim: keep only the longest stretch that is a clean parabola (drops the throw /
            catch / bounce at the ends).

    Returns:
        dict with ``g`` (and ``g_err``, standard error), ``launch_speed``, ``launch_angle_deg``
        (above horizontal, at the first fitted frame), ``apex_height``, ``flight_time_s``,
        ``r2``, ``rms_residual_px``, ``start_s``/``end_s`` of the fitted part, the units, and
        ``fit_frames`` / ``fit_x`` / ``fit_y`` (fitted curve in pixels) for plotting.
    """
    center = np.asarray(center, dtype=np.float64)
    T = len(center)
    valid = np.all(np.isfinite(center), axis=1) & _window_mask(T, fps, window)
    s, e = _longest_run(valid)
    if e - s < 6:
        raise ValueError("Projectile fit needs the object visible for at least 6 consecutive frames")
    frames = np.arange(s, e)
    t, x, y = frames / fps, center[s:e, 0], center[s:e, 1]
    keep = _robust_flight(t, x, y) if auto_trim else np.ones(len(t), bool)
    if keep.sum() < 6:
        raise ValueError("Could not find a clean flight section; set the time window manually")
    frames, t, x, y = frames[keep], t[keep], x[keep], y[keep]
    t0 = t[0]
    tt = t - t0

    py, cov_y = np.polyfit(tt, y, 2, cov=True)               # y = a tt^2 + b tt + c, a = g/2
    px_ = np.polyfit(tt, x, 1)                               # x = vx tt + x0
    a, b, _c = py
    vx, vy = px_[0], b                                       # px/s at the first fitted frame
    scale = m_per_px or 1.0
    unit_len = "m" if m_per_px else "px"

    fit_y = np.polyval(py, tt)
    fit_x = np.polyval(px_, tt)
    ss_res = np.sum((y - fit_y) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    res = np.hypot(fit_x - x, fit_y - y)

    g = 2 * a * scale
    g_err = 2 * np.sqrt(max(cov_y[0, 0], 0.0)) * scale
    # apex: vertical velocity zero (image y is down, so "up" is negative vy)
    t_apex = -b / (2 * a) if a != 0 else np.nan
    apex_rise = (y[0] - np.polyval(py, t_apex)) * scale if 0 <= t_apex <= tt[-1] * 2 else float("nan")
    out = {
        "model": "projectile",
        "g": float(g), "g_err": float(g_err), "unit_g": f"{unit_len}/s^2",
        "g_error_vs_standard_pct": float((g - G_STANDARD) / G_STANDARD * 100) if m_per_px else None,
        "launch_speed": float(np.hypot(vx, vy) * scale), "unit_speed": f"{unit_len}/s",
        # direction is undefined for an object released from rest (free fall)
        "launch_angle_deg": (float(np.degrees(np.arctan2(-vy, abs(vx))))
                             if np.hypot(vx, vy) > 0.1 * abs(a) * max(tt[-1], 1e-9) else None),
        "horizontal_speed": float(abs(vx) * scale),
        "apex_height": float(apex_rise), "apex_time_s": float(t0 + t_apex) if np.isfinite(t_apex) else None,
        "unit_length": unit_len, "m_per_px": float(m_per_px) if m_per_px else None,
        "flight_time_s": float(tt[-1]),
        "start_s": float(t0), "end_s": float(t[-1]),
        "frames_used": int(len(t)),
        "r2": float(1 - ss_res / ss_tot), "rms_residual_px": float(np.sqrt(np.mean(res ** 2))),
        "fit_frames": frames, "fit_x": fit_x, "fit_y": fit_y,
    }
    return out


def _damped_cos(t, A, tau, period, phi, c):
    return c + A * np.exp(-t / tau) * np.cos(2 * np.pi * t / period + phi)


def fit_pendulum(center: np.ndarray, fps: float, length_m: Optional[float] = None,
                 window: Optional[Sequence[float]] = None, axis: Optional[int] = None) -> dict:
    """Fit a (damped) oscillation to an object path, e.g. a pendulum bob or a spring.

    Args:
        center: (T, 2) positions in pixels (NaN where not visible).
        length_m: pendulum length (pivot to the centre of the bob) in metres; gives ``g``.
        axis: 0 = horizontal, 1 = vertical motion; default: whichever varies more.

    Returns:
        dict with ``period_s`` (``period_err_s``), ``frequency_hz``, ``amplitude_px``,
        ``damping_time_s``, ``oscillations``, ``r2``, ``g`` / ``g_err`` when ``length_m`` is
        given, and ``fit_frames`` / ``fit_values`` for plotting.
    """
    center = np.asarray(center, dtype=np.float64)
    T = len(center)
    valid = np.all(np.isfinite(center), axis=1) & _window_mask(T, fps, window)
    s, e = _longest_run(valid)
    if e - s < 12:
        raise ValueError("Pendulum fit needs the object visible for at least 12 consecutive frames")
    frames = np.arange(s, e)
    t = (frames - s) / fps
    if axis is None:
        axis = int(np.argmax(np.nanstd(center[s:e], axis=0)))
    v = center[s:e, axis]

    # initial guess: dominant frequency from the spectrum
    d = v - v.mean()
    spec = np.abs(np.fft.rfft(d * np.hanning(len(d))))
    freqs = np.fft.rfftfreq(len(d), 1 / fps)
    spec[0] = 0
    k = int(np.argmax(spec))
    if k == 0 or freqs[k] <= 0:
        raise ValueError("No oscillation found in the path")
    p0 = [d.std() * np.sqrt(2), t[-1] * 5 + 1, 1 / freqs[k], 0.0, v.mean()]
    best = None
    for phi in np.linspace(-np.pi, np.pi, 8, endpoint=False):   # avoid a bad local minimum
        p0[3] = phi
        try:
            popt, pcov = curve_fit(_damped_cos, t, v, p0=p0, maxfev=20000,
                                   bounds=([0, 0.05, 0.5 / freqs[k], -10, -np.inf],
                                           [np.inf, 1e6, 2 / freqs[k], 10, np.inf]))
        except RuntimeError:
            continue
        sse = np.sum((_damped_cos(t, *popt) - v) ** 2)
        if best is None or sse < best[0]:
            best = (sse, popt, pcov)
    if best is None:
        raise ValueError("Pendulum fit did not converge")
    sse, popt, pcov = best
    A, tau, period, _phi, _c = popt
    period_err = float(np.sqrt(max(pcov[2, 2], 0.0)))
    fit = _damped_cos(t, *popt)
    ss_tot = np.sum((v - v.mean()) ** 2) + 1e-12
    out = {
        "model": "pendulum",
        "period_s": float(period), "period_err_s": period_err, "frequency_hz": float(1 / period),
        "amplitude_px": float(abs(A)), "damping_time_s": float(tau),
        "oscillations": float(t[-1] / period), "axis": "horizontal" if axis == 0 else "vertical",
        "r2": float(1 - sse / ss_tot), "start_s": float(s / fps), "end_s": float((e - 1) / fps),
        "fit_frames": frames, "fit_values": fit, "fit_axis": axis,
    }
    if length_m:
        g = 4 * np.pi ** 2 * length_m / period ** 2
        out.update(g=float(g), g_err=float(2 * g * period_err / period), unit_g="m/s^2",
                   g_error_vs_standard_pct=float((g - G_STANDARD) / G_STANDARD * 100),
                   length_m=float(length_m))
    return out


def public_summary(fit: dict) -> dict:
    """The fit result without the per-frame arrays (for JSON)."""
    return {k: v for k, v in fit.items() if not isinstance(v, np.ndarray)}

import json

import numpy as np
import pytest

from tracker import fit_pendulum, fit_projectile, track_video
from tracker.cli import main as cli_main

G = 9.81


def throw_path(fps=60.0, m_per_px=0.005, vx=2.0, vy_up=3.0, hold=20, land=20, noise=0.4, seed=0):
    """Ball held still, thrown, then lying on the ground. Image coords (y down), pixels."""
    rng = np.random.default_rng(seed)
    t_flight = 2 * vy_up / G                                          # back to launch height
    tf = np.arange(0, t_flight, 1 / fps)
    x = 100 + vx * tf / m_per_px
    y = 300 - (vy_up * tf - G * tf ** 2 / 2) / m_per_px
    path = np.concatenate([np.tile([x[0], y[0]], (hold, 1)), np.stack([x, y], -1), np.tile([x[-1], y[-1]], (land, 1))])
    return path + rng.normal(0, noise, path.shape), hold / fps


def test_projectile_recovers_g_speed_and_angle():
    path, t0 = throw_path()
    p = fit_projectile(path, 60.0, m_per_px=0.005)
    assert p["g"] == pytest.approx(G, rel=0.02)
    assert p["g_err"] < 0.2
    assert p["launch_speed"] == pytest.approx(np.hypot(2, 3), rel=0.03)
    assert p["launch_angle_deg"] == pytest.approx(np.degrees(np.arctan2(3, 2)), abs=1.5)
    assert p["start_s"] == pytest.approx(t0, abs=0.05)               # the hold was trimmed away
    assert p["r2"] > 0.999 and p["unit_g"] == "m/s^2"
    assert abs(p["g_error_vs_standard_pct"]) < 2


def test_free_fall_and_pixel_units():
    fps = 120.0
    t = np.arange(0, 0.6, 1 / fps)
    path = np.stack([np.full_like(t, 50.0), 20 + 0.5 * 980.0 * t ** 2], -1)   # 980 px/s^2
    p = fit_projectile(path, fps)
    assert p["unit_g"] == "px/s^2" and p["g"] == pytest.approx(980, rel=1e-3)
    # dropped: the fit starts once it is clearly moving, i.e. falling straight down
    assert p["launch_angle_deg"] is None or p["launch_angle_deg"] == pytest.approx(-90, abs=1)
    assert p["g_error_vs_standard_pct"] is None


def test_projectile_window_and_errors():
    path, _ = throw_path()
    p = fit_projectile(path, 60.0, 0.005, window=(0.5, 0.8), auto_trim=False)
    assert p["start_s"] >= 0.5 and p["end_s"] <= 0.8
    with pytest.raises(ValueError):
        fit_projectile(np.full((5, 2), 1.0), 30.0)


def pendulum_path(L=0.5, fps=60.0, seconds=8.0, amp=60.0, tau=25.0, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / fps)
    T = 2 * np.pi * np.sqrt(L / G)
    x = 320 + amp * np.exp(-t / tau) * np.cos(2 * np.pi * t / T + 0.7)
    y = 200 + 4 * np.cos(4 * np.pi * t / T)                          # small vertical bob motion
    return np.stack([x, y], -1) + rng.normal(0, noise, (len(t), 2)), T


def test_pendulum_period_and_g():
    path, T = pendulum_path()
    p = fit_pendulum(path, 60.0, length_m=0.5)
    assert p["period_s"] == pytest.approx(T, rel=0.002)
    assert p["g"] == pytest.approx(G, rel=0.005)
    assert p["axis"] == "horizontal" and p["oscillations"] > 5 and p["r2"] > 0.99


def test_pendulum_without_length_and_occlusion():
    path, T = pendulum_path(L=1.0)
    path[100:103] = np.nan                                            # hidden briefly: longest run is used
    p = fit_pendulum(path, 60.0)
    assert "g" not in p and p["period_s"] == pytest.approx(T, rel=0.01)
    with pytest.raises(ValueError):
        fit_pendulum(np.full((10, 2), 1.0), 30.0)


def _throw_tracker(frames, queries):
    T = frames.shape[0]
    path, _ = throw_path(fps=30.0, hold=5, land=T, noise=0.0)
    path = path[:T] * 0.5                                            # fits a 320x180 video
    tracks = np.repeat(path[:, None, :], len(queries), axis=1).astype(np.float32)
    return tracks, np.ones((T, len(queries)), np.float32)


def test_track_video_physics_end_to_end(make_video, tmp_path):
    out = tmp_path / "o"
    r = track_video(make_video(width=320, height=180, frames=40, fps=30.0), points=[(50, 150)],
                    physics="projectile", reference=(0, 0, 100, 0, 1.0),      # 100 px = 1 m
                    output_path=str(out / "v.mp4"), json_path=str(out / "s.json"),
                    max_side=None, tracker_fn=_throw_tracker)
    # the path was built with 0.005 m/px and then halved -> 0.01 m/px; we told it 0.01 m/px
    assert r.physics["g"] == pytest.approx(G, rel=0.02)
    s = json.load(open(out / "s.json", encoding="utf-8"))
    assert s["physics"]["model"] == "projectile" and "fit_x" not in s["physics"]
    with pytest.raises(ValueError):
        track_video(make_video(), points=[(20, 90)], physics="rocket", tracker_fn=_throw_tracker)


def test_cli_physics_arguments(tmp_path):
    with pytest.raises(SystemExit):
        cli_main(["--video", "x.mp4", "--point", "1,2", "--physics", "rocket"])
    assert cli_main(["--video", str(tmp_path / "none.mp4"), "--point", "1,2", "--physics", "pendulum",
                     "--pendulum-length", "0.5", "--window", "0,3"]) == 1

import csv
import json

import numpy as np
import pytest

from tracker import angle_at, joint_angle_series, repetition_stats, track_video
from tracker.cli import main as cli_main


def arm_angles(T=180, fps=30.0, reps=3, lo=90.0, hi=170.0):
    """Knee-like motion: straight (hi) -> bent (lo) -> straight, ``reps`` times."""
    t = np.arange(T) / fps
    phase = 2 * np.pi * reps * t / (T / fps)
    return (hi + lo) / 2 + (hi - lo) / 2 * np.cos(phase)


def limb_points(angles_deg, upper=80.0, lower=70.0, joint=(150.0, 100.0)):
    """Hip above the knee, ankle rotated by the joint angle: (T, 3, 2) points end, joint, end."""
    T = len(angles_deg)
    jx, jy = joint
    hip = np.tile([jx, jy - upper], (T, 1))
    a = np.radians(angles_deg)
    # angle between (hip - knee) = straight up and (ankle - knee)
    ankle = np.stack([jx + lower * np.sin(a), jy - lower * np.cos(a)], -1)
    knee = np.tile([jx, jy], (T, 1))
    return np.stack([hip, knee, ankle], 1)


def test_angle_at_basic():
    assert angle_at(np.array([1.0, 0]), np.array([0.0, 0]), np.array([0.0, 1])) == pytest.approx(90)
    assert angle_at(np.array([-1.0, 0]), np.array([0.0, 0]), np.array([1.0, 0])) == pytest.approx(180)
    assert np.isnan(angle_at(np.array([0.0, 0]), np.array([0.0, 0]), np.array([1.0, 0])))


def test_joint_angle_series_matches_geometry_and_hides_invisible():
    truth = arm_angles()
    pts = limb_points(truth)
    vis = np.ones((len(truth), 3))
    vis[50:55, 2] = 0
    ang = joint_angle_series(pts, vis, (0, 1, 2))
    ok = np.isfinite(ang)
    np.testing.assert_allclose(ang[ok], truth[ok], atol=1e-6)
    assert not ok[50:55].any() and ok.sum() == len(truth) - 5


def test_repetition_stats_counts_reps_and_range():
    rng = np.random.default_rng(0)
    ang = arm_angles(reps=3) + rng.normal(0, 1.5, 180)           # tracking jitter
    s = repetition_stats(ang, 30.0)
    assert s["reps"] == 3
    assert s["rom_deg"] == pytest.approx(80, abs=5)
    assert s["min_deg"] == pytest.approx(90, abs=4) and s["max_deg"] == pytest.approx(170, abs=4)
    assert s["mean_rep_s"] == pytest.approx(2.0, abs=0.2)        # 6 s / 3 reps
    assert all(e["kind"] == "most bent" for e in s["rep_events"])


def test_repetition_stats_ignores_wobble_and_handles_gaps():
    rng = np.random.default_rng(1)
    still = 150 + rng.normal(0, 2, 120)                            # holding still: no reps
    assert repetition_stats(still, 30.0)["reps"] == 0
    ang = arm_angles(reps=4, T=240)
    ang[100:104] = np.nan                                          # short occlusion is bridged
    assert repetition_stats(ang, 30.0)["reps"] == 4
    assert repetition_stats(np.full(10, np.nan), 30.0)["reps"] == 0


def _limb_tracker(frames, queries):
    T = frames.shape[0]
    pts = limb_points(arm_angles(T=T, reps=2))
    tracks = np.repeat(pts[:, :1], len(queries), axis=1).astype(np.float32)
    tracks[:, :3] = pts                                            # the three joint points
    return tracks, np.ones((T, len(queries)), np.float32)


def test_track_video_joints_end_to_end(make_video, tmp_path):
    out = tmp_path / "o"
    r = track_video(make_video(width=320, height=180, frames=60, fps=30.0),
                    joints=[[(150, 20), (150, 100), (150, 170)]], joint_names=["knee"],
                    output_path=str(out / "v.mp4"), csv_path=str(out / "p.csv"), json_path=str(out / "s.json"),
                    max_side=None, smooth=False, tracker_fn=_limb_tracker)
    j = r.joints[0]
    assert j["name"] == "knee" and j["reps"] == 2
    assert j["rom_deg"] == pytest.approx(80, abs=3)
    rows = list(csv.DictReader(open(out / "p_joints.csv", encoding="utf-8")))
    assert len(rows) == 60 and "knee_deg" in rows[0]
    assert float(rows[0]["knee_deg"]) == pytest.approx(170, abs=0.5)
    s = json.load(open(out / "s.json", encoding="utf-8"))
    assert s["joints"][0]["reps"] == 2


def test_joints_with_box_keep_box_math_separate(make_video):
    from conftest import fake_tracker
    r = track_video(make_video(), joints=[[(20, 80), (40, 90), (60, 80)]], box=(20, 80, 60, 100),
                    tracker_fn=fake_tracker)
    assert r.boxes.shape == (40, 4) and len(r.object_points) == 16 and len(r.queries) == 19


def test_joint_needs_three_points(make_video):
    from conftest import fake_tracker
    with pytest.raises(ValueError):
        track_video(make_video(), joints=[[(1, 1), (2, 2)]], tracker_fn=fake_tracker)


def test_cli_joint_argument(tmp_path):
    with pytest.raises(SystemExit):
        cli_main(["--video", "x.mp4", "--joint", "1,2,3,4"])
    assert cli_main(["--video", str(tmp_path / "none.mp4"), "--joint", "1,2,3,4,5,6"]) == 1

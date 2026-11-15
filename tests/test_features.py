import csv
import json

import cv2
import numpy as np
import pytest

from conftest import fake_tracker
from tracker import (apply_transforms, boxes_from_tracks, object_point_mask, estimate_camera_motion, heatmap, line_crossings,
                     meters_per_pixel, motion_stats, object_center, points_in_box, track_video, zone_events)
from tracker.cli import main as cli_main


# ---------------------------------------------------------------- selection
def test_points_in_box_inside_with_margin():
    pts = points_in_box((10, 20, 110, 70), grid=4, margin=0.1)
    assert len(pts) == 16
    xs, ys = zip(*pts)
    assert min(xs) == pytest.approx(20) and max(xs) == pytest.approx(100)
    assert min(ys) == pytest.approx(25) and max(ys) == pytest.approx(65)
    assert points_in_box((0, 0, 10, 10), grid=1) == [(5.0, 5.0)]
    with pytest.raises(ValueError):
        points_in_box((5, 5, 6, 6))


def test_box_follows_translation_and_scale():
    p0 = np.array(points_in_box((40, 40, 60, 60), grid=3), np.float32)       # centre (50, 50)
    # frames 0-9: moved by (10, 5); frames 10-19: scaled x2 around (80, 60)
    tracks = np.stack([p0] + [p0 + [10, 5]] * 9 + [(p0 - 50) * 2 + [80, 60]] * 10)
    boxes = boxes_from_tracks(tracks, np.ones((20, len(p0))), (40, 40, 60, 60), query_frame=0)
    np.testing.assert_allclose(boxes[0], [40, 40, 60, 60], atol=1e-4)
    np.testing.assert_allclose(boxes[5], [50, 45, 70, 65], atol=1e-4)
    np.testing.assert_allclose(boxes[19], [60, 40, 100, 80], atol=1e-4)


def test_box_ignores_drifting_point_and_hidden_frames():
    p0 = np.array(points_in_box((0, 0, 100, 100), grid=3), np.float32)
    tracks = np.stack([p0, p0 + 10])
    tracks[1, 0] = [900, 900]                                                 # one point drifts away
    vis = np.ones((2, 9))
    mask = object_point_mask(tracks, vis, (0, 0, 100, 100), 0)
    assert not mask[0] and mask[1:].all()
    boxes = boxes_from_tracks(tracks, vis, (0, 0, 100, 100), 0, point_mask=mask)
    np.testing.assert_allclose(boxes[1], [10, 10, 110, 110], atol=1.0)
    vis[1] = 0
    assert np.isnan(boxes_from_tracks(tracks, vis, (0, 0, 100, 100), 0, point_mask=mask)[1]).all()


def test_box_stays_centred_when_points_rotate_out_of_view():
    # object fixed at x in [0, 100]; its surface points slide right and the left ones become hidden
    p0 = np.array(points_in_box((0, 0, 100, 100), grid=4), np.float32)
    T = 10
    tracks = np.repeat(p0[None], T, 0)
    vis = np.ones((T, 16))
    for t in range(1, T):
        tracks[t, :, 0] = np.clip(p0[:, 0] + 3 * t, 0, 100)
        vis[t, tracks[t, :, 0] >= 100] = 0          # rotated to the back
    tracks[T - 1, :, 0] = np.where(vis[T - 1] > 0, tracks[T - 1, :, 0], 50)   # model's guess for hidden ones
    boxes = boxes_from_tracks(tracks, vis, (0, 0, 100, 100), 0)
    assert abs((boxes[T - 1, 0] + boxes[T - 1, 2]) / 2 - 50) < 25


def test_object_center_median_and_nan():
    tracks = np.array([[[0, 0], [2, 2], [100, 100]], [[5, 5], [5, 5], [5, 5]]], np.float32)
    vis = np.array([[1, 1, 0], [0, 0, 0]])
    c = object_center(tracks, vis)
    np.testing.assert_allclose(c[0], [1, 1])
    assert np.isnan(c[1]).all()


# ---------------------------------------------------------------- analysis
def test_meters_per_pixel():
    assert meters_per_pixel((0, 0), (100, 0), 2.0) == pytest.approx(0.02)
    with pytest.raises(ValueError):
        meters_per_pixel((0, 0), (0, 0), 1.0)


def test_motion_stats_constant_speed():
    fps = 25.0
    center = np.stack([np.arange(50) * 4.0, np.zeros(50)], -1)              # 4 px/frame = 100 px/s
    m = motion_stats(center, fps)
    assert m["unit_speed"] == "px/s"
    assert m["avg_speed"] == pytest.approx(100, rel=1e-3)
    assert m["distance"] == pytest.approx(196, rel=1e-3)
    mc = motion_stats(center, fps, m_per_px=0.05)                           # 5 m/s = 18 km/h
    assert mc["unit_speed"] == "km/h" and mc["unit_distance"] == "m"
    assert mc["avg_speed"] == pytest.approx(18, rel=1e-3)
    assert mc["distance"] == pytest.approx(9.8, rel=1e-3)


def test_motion_stats_skips_gaps():
    center = np.stack([np.arange(20) * 1.0, np.zeros(20)], -1)
    center[10:12] = np.nan
    m = motion_stats(center, 10.0, smooth=False)
    assert m["distance"] == pytest.approx(16)                                # 19 steps minus 3 touching the gap
    assert np.isnan(m["speed"][10:13]).all()


def test_line_crossings_both_directions():
    xs = np.concatenate([np.arange(0, 21), np.arange(20, -1, -1)]).astype(float)
    center = np.stack([xs, np.full_like(xs, 5)], -1)
    ev = line_crossings(center, (10.5, 0, 10.5, 10), fps=10)
    assert [e["direction"] for e in ev] == ["B->A", "A->B"]
    assert ev[0]["frame"] == 11
    # landing exactly on the line for a frame still counts once
    on = np.stack([np.arange(0, 21, dtype=float), np.full(21, 5.0)], -1)
    assert len(line_crossings(on, (10, 0, 10, 10), fps=10)) == 1
    # path outside the segment's extent does not count
    assert line_crossings(center, (10.5, 20, 10.5, 30), fps=10) == []


def test_zone_events():
    xs = np.arange(0, 30, dtype=float)
    center = np.stack([xs, np.full_like(xs, 5)], -1)
    z = zone_events(center, [(10, 0), (20, 0), (20, 10), (10, 10)], fps=10)
    assert [e["frame"] for e in z["entries"]] == [10]
    assert [e["frame"] for e in z["exits"]] == [21]
    assert z["time_inside_s"] == pytest.approx(1.1)


def test_heatmap_peak_where_object_stays():
    pts = np.array([[20, 30]] * 50 + [[80, 10]] * 5, float)
    h = heatmap(pts, (100, 50))
    assert h.shape == (50, 100) and h.max() == pytest.approx(1.0)
    assert np.unravel_index(h.argmax(), h.shape) == (30, 20)


# ---------------------------------------------------------------- stabilisation
def _textured(h=120, w=160, seed=0):
    rng = np.random.default_rng(seed)
    img = (rng.random((h // 4, w // 4)) * 255).astype(np.uint8)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(img, (3, 3), 0)


def test_camera_motion_recovers_pan():
    base = _textured(240, 320)
    T, shift = 6, 3                                                           # camera pans 3 px/frame right
    frames = np.stack([cv2.cvtColor(base[40:160, 40 + shift * t: 200 + shift * t], cv2.COLOR_GRAY2RGB)
                       for t in range(T)])
    tf = estimate_camera_motion(frames, reference_frame=0)
    # a static world point appears 3 px further left every frame; compensation maps it back
    pts = np.stack([np.array([100.0 - shift * t, 60.0]) for t in range(T)])
    np.testing.assert_allclose(apply_transforms(pts, tf), np.tile([100, 60], (T, 1)), atol=0.5)


# ---------------------------------------------------------------- pipeline with all features
def test_track_video_box_and_measurements(make_video, tmp_path):
    out = tmp_path / "o"
    r = track_video(make_video(width=320, height=180, fps=20.0), box=(20, 80, 60, 100), query_frame=0,
                    output_path=str(out / "v.mp4"), csv_path=str(out / "p.csv"), json_path=str(out / "s.json"),
                    heatmap_path=str(out / "h.png"), max_side=None,
                    reference=(0, 0, 100, 0, 1.0),           # 100 px = 1 m
                    count_lines=[(100, 0, 100, 180)], zones=[[(110, 0), (320, 0), (320, 180), (110, 180)]],
                    tracker_fn=fake_tracker)
    assert r.boxes.shape == (40, 4) and len(r.queries) == 16
    # fake tracker: 2 px/frame at 20 fps = 40 px/s = 0.4 m/s = 1.44 km/h
    assert r.motion["unit_speed"] == "km/h"
    assert r.motion["avg_speed"] == pytest.approx(1.44, rel=0.02)
    assert len(r.crossings) == 1 and r.crossings[0]["direction"] in ("A->B", "B->A")
    assert r.zones[0]["entries"]

    s = json.load(open(out / "s.json", encoding="utf-8"))
    assert s["box_mode"] and s["num_points"] == 16 and s["line_crossings"]
    rows = list(csv.DictReader(open(out / "p_object.csv", encoding="utf-8")))
    assert len(rows) == 40 and "speed_km_per_h" in rows[0] and "box_x1" in rows[0]
    assert rows[12]["visible"] == "0" and rows[12]["center_x"] == ""
    assert cv2.imread(str(out / "h.png")).shape == (180, 320, 3)
    assert int(cv2.VideoCapture(str(out / "v.mp4")).get(cv2.CAP_PROP_FRAME_COUNT)) == 40


def test_track_video_stabilize_runs(make_video):
    r = track_video(make_video(), points=[(20, 90)], stabilize=True, tracker_fn=fake_tracker)
    assert r.stabilized and r.center.shape == (40, 2)


def test_browser_video_is_h264(make_video, tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    out = str(tmp_path / "b.mp4")
    track_video(make_video(), points=[(20, 90)], output_path=out, browser_compatible=True, tracker_fn=fake_tracker)
    cap = cv2.VideoCapture(out)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="ignore")
    assert fourcc.lower() in ("avc1", "h264")


def test_cli_requires_point_or_box():
    with pytest.raises(SystemExit):
        cli_main(["--video", "x.mp4"])


def test_cli_parses_zone_and_reference(tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli_main(["--video", "x.mp4", "--point", "1,2", "--zone", "1,2;3,4"])
    assert cli_main(["--video", str(tmp_path / "none.mp4"), "--box", "0,0,10,10",
                     "--reference", "0,0,10,0,1", "--zone", "0,0;5,0;5,5"]) == 1


def test_object_reference_scale_follows_object_size():
    from tracker import object_scale_m_per_px
    T = 30
    w = np.linspace(100, 200, T)                       # object doubles in size (comes closer)
    boxes = np.stack([np.zeros(T), np.zeros(T), w, w / 2], -1)
    mpp = object_scale_m_per_px(0.04, boxes, 0, smooth_frames=1)
    assert mpp[0] == pytest.approx(0.04) and mpp[-1] == pytest.approx(0.02)
    # camera zoomed in 2x while positions were mapped to the reference frame -> cancels out
    mpp_z = object_scale_m_per_px(0.04, boxes, 0, zoom=np.linspace(1, 0.5, T), smooth_frames=1)
    assert mpp_z[-1] == pytest.approx(0.04)


def test_motion_stats_per_frame_scale():
    center = np.stack([np.arange(11) * 10.0, np.zeros(11)], -1)   # 10 px/frame
    m = motion_stats(center, 1.0, m_per_px=np.full(11, 0.1), smooth=False)
    assert m["distance"] == pytest.approx(10.0)                  # 100 px * 0.1 m
    assert m["avg_speed"] == pytest.approx(3.6)                  # 1 m/s


def test_reference_is_object_requires_box(make_video):
    with pytest.raises(ValueError):
        track_video(make_video(), points=[(20, 90)], reference=(0, 0, 10, 0, 1.0),
                    reference_is_object=True, tracker_fn=fake_tracker)

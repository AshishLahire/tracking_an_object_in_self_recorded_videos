import csv
import json

import cv2
import numpy as np
import pytest

from tracker import crossing_times, speed_trap, track_objects
from tracker.cli import main as cli_main


def straight(T, x0, vx, y=50.0):
    """Path moving vx px/frame along x."""
    t = np.arange(T, dtype=float)
    return np.stack([x0 + vx * t, np.full(T, y)], -1)


# ---------------------------------------------------------------- crossing timing
def test_crossing_time_is_interpolated_between_frames():
    path = straight(40, 10.0, 7.3)                    # reaches x = 100 at frame 90/7.3 = 12.33
    ev = crossing_times(path, (100, 0, 100, 100), fps=30.0)
    assert len(ev) == 1 and ev[0]["direction"] == "B->A"
    assert ev[0]["time_s"] == pytest.approx((90 / 7.3) / 30.0, abs=1e-6)


def test_crossing_outside_segment_or_across_long_gap_is_ignored():
    path = straight(40, 10.0, 7.3)
    assert crossing_times(path, (100, 60, 100, 100), 30.0) == []          # line does not reach the path
    gap = path.copy()
    gap[10:30] = np.nan                                                    # hidden for 0.67 s around the line
    assert crossing_times(gap, (100, 0, 100, 100), 30.0) == []


# ---------------------------------------------------------------- speed trap
def test_speed_trap_speed_direction_and_limit():
    fps = 30.0
    # 5 px/frame; lines 200 px apart = 40 frames = 1.333 s; the real distance is 20 m -> 15 m/s = 54 km/h
    path = straight(120, 0.0, 5.0)
    passes = speed_trap(path, (100, 0, 100, 100), (300, 0, 300, 100), 20.0, fps, limit_kmh=50)
    assert len(passes) == 1
    p = passes[0]
    assert (p["from"], p["to"]) == ("A", "B")
    assert p["seconds"] == pytest.approx(40 / fps, abs=1e-4)
    assert p["speed_kmh"] == pytest.approx(54.0, abs=0.01) and p["over_limit"]


def test_speed_trap_both_directions_and_laps():
    fps = 25.0
    there = straight(60, 0.0, 10.0)                                        # A -> B at 10 px/frame
    back = straight(120, 590.0, -5.0)                                      # B -> A at 5 px/frame
    passes = speed_trap(np.concatenate([there, back]), (100, 0, 100, 100), (400, 0, 400, 100), 30.0, fps)
    assert [(p["from"], p["to"]) for p in passes] == [("A", "B"), ("B", "A")]
    assert passes[0]["speed_kmh"] == pytest.approx(30 / (30 / fps) * 3.6, rel=1e-3)    # 30 frames
    assert passes[1]["speed_kmh"] == pytest.approx(30 / (60 / fps) * 3.6, rel=1e-3)    # 60 frames
    assert not any(p["over_limit"] for p in passes)


def test_speed_trap_needs_both_lines_and_positive_distance():
    path = straight(30, 0.0, 5.0)                                          # only reaches x = 145
    assert speed_trap(path, (100, 0, 100, 100), (300, 0, 300, 100), 20.0, 30.0) == []
    with pytest.raises(ValueError):
        speed_trap(path, (100, 0, 100, 100), (300, 0, 300, 100), 0.0, 30.0)


# ---------------------------------------------------------------- multi-object pipeline
def two_car_video(tmp_path, T=60, fps=30.0):
    path = str(tmp_path / "cars.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (400, 200))
    for _ in range(T):
        w.write(np.zeros((200, 400, 3), np.uint8))
    w.release()
    return path


def fake_two_cars(frames, queries):
    """Points with y < 100 (top lane) move 4 px/frame right, the others 8 px/frame left."""
    T = frames.shape[0]
    t = np.arange(T, dtype=np.float32)[:, None]
    v = np.where(queries[:, 2] < 100, 4.0, -8.0)[None, :]
    tracks = np.stack([queries[None, :, 1] + v * t, np.repeat(queries[None, :, 2], T, 0)], -1)
    return tracks.astype(np.float32), np.ones((T, len(queries)), np.float32)


def test_track_objects_two_cars_speed_trap(tmp_path):
    out = tmp_path / "o"
    trap = {"line_a": (120, 0, 120, 200), "line_b": (220, 0, 220, 200), "distance_m": 10.0, "limit_kmh": 60}
    r = track_objects(two_car_video(tmp_path), [{"box": (10, 30, 50, 60), "name": "slow car"},
                                                {"box": (340, 130, 390, 170), "name": "fast car"}],
                      speed_trap=trap, output_path=str(out / "v.mp4"), csv_path=str(out / "o.csv"),
                      json_path=str(out / "s.json"), max_side=None, tracker_fn=fake_two_cars)
    by_name = {p["object"]: p for p in r.passes}
    # 100 px between lines: slow 4 px/f -> 25 frames = 0.833 s -> 43.2 km/h; fast 8 px/f -> 0.4167 s -> 86.4 km/h
    assert by_name["slow car"]["speed_kmh"] == pytest.approx(43.2, abs=0.05)
    assert (by_name["slow car"]["from"], by_name["fast car"]["from"]) == ("A", "B")
    assert by_name["fast car"]["speed_kmh"] == pytest.approx(86.4, abs=0.05)
    assert by_name["fast car"]["over_limit"] and not by_name["slow car"]["over_limit"]

    s = json.load(open(out / "s.json", encoding="utf-8"))
    assert s["speed_trap"]["passes"] == 2 and s["speed_trap"]["over_limit"] == 1
    rows = list(csv.DictReader(open(out / "o_speed_trap.csv", encoding="utf-8")))
    assert [r_["object"] for r_ in rows] == [p["object"] for p in r.passes]
    obj_rows = list(csv.DictReader(open(out / "o.csv", encoding="utf-8")))
    assert len(obj_rows) == 60 * 2 and {"slow car", "fast car"} == {x["object"] for x in obj_rows}
    assert int(cv2.VideoCapture(str(out / "v.mp4")).get(cv2.CAP_PROP_FRAME_COUNT)) == 60


def test_track_objects_point_object_and_validation(tmp_path):
    video = two_car_video(tmp_path)
    r = track_objects(video, [[(30, 40), (35, 45)]], count_lines=[(100, 0, 100, 200)], max_side=None,
                      tracker_fn=fake_two_cars)
    assert r.objects[0].boxes is None and len(r.objects[0].crossings) == 1
    with pytest.raises(ValueError):
        track_objects(video, [], tracker_fn=fake_two_cars)
    with pytest.raises(ValueError):
        track_objects(video, [(10, 30, 50, 60)], speed_trap={"line_a": (0, 0, 1, 1)}, tracker_fn=fake_two_cars)


def test_cli_multi_arguments(tmp_path):
    with pytest.raises(SystemExit):
        cli_main(["--video", "x.mp4", "--object", "1,2,3,4", "--trap", "1,2,3,4,5"])
    assert cli_main(["--video", str(tmp_path / "none.mp4"), "--object", "1,2,30,40", "--object", "5,6,70,80",
                     "--trap", "10,0,10,100,50,0,50,100,12.5", "--speed-limit", "50"]) == 1

import csv

import cv2
import numpy as np
import pytest

from conftest import fake_tracker
from tracker import (evaluate_tracking, load_video, make_queries, point_colors, render_frames,
                     smooth_tracks, track_video, visible_runs)
from tracker.cli import main as cli_main


# ---------------------------------------------------------------- video_io
def test_load_video_keeps_fps_and_size(make_video):
    v = load_video(make_video(fps=25.0), max_side=None)
    assert v.frames.shape == (40, 180, 320, 3)
    assert v.frames.dtype == np.uint8
    assert v.fps == pytest.approx(25.0, abs=0.01)
    assert v.scale == 1.0 and v.original_size == (320, 180)


def test_load_video_downscales_to_even_size(make_video):
    v = load_video(make_video(width=850, height=478), max_side=640)
    w, h = v.size
    assert max(w, h) <= 640 and w % 2 == 0 and h % 2 == 0
    assert v.scale == pytest.approx(640 / 850)


def test_load_video_max_frames(make_video):
    assert load_video(make_video(), max_frames=7).num_frames == 7


def test_load_video_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_video(str(tmp_path / "nope.mp4"))


# ---------------------------------------------------------------- smoothing
def test_visible_runs():
    assert visible_runs([1, 1, 0, 0, 1, 0, 1, 1, 1]) == [(0, 2), (4, 5), (6, 9)]
    assert visible_runs([0, 0]) == []
    assert visible_runs([1, 1, 1]) == [(0, 3)]


def test_smoothing_ignores_occluded_frames():
    T = 40
    tracks = np.zeros((T, 1, 2), np.float32)
    tracks[:, 0, 0] = np.arange(T)
    tracks[18:22, 0] = 1000.0                 # garbage while occluded
    vis = np.ones((T, 1)); vis[18:22] = 0
    out = smooth_tracks(tracks, vis)
    # visible parts are a straight line -> unchanged by Savitzky-Golay, and not pulled toward 1000
    np.testing.assert_allclose(out[:18, 0, 0], np.arange(18), atol=1e-3)
    np.testing.assert_allclose(out[22:, 0, 0], np.arange(22, T), atol=1e-3)
    np.testing.assert_allclose(out[18:22, 0], 1000.0)


def test_smoothing_reduces_noise():
    rng = np.random.default_rng(0)
    clean = np.stack([np.linspace(0, 100, 60), np.linspace(0, 50, 60)], -1)[:, None]
    noisy = (clean + rng.normal(0, 3, clean.shape)).astype(np.float32)
    out = smooth_tracks(noisy, np.ones((60, 1)))
    assert np.abs(out - clean).mean() < np.abs(noisy - clean).mean()


# ---------------------------------------------------------------- draw
def test_point_colors_distinct_and_bright():
    cols = point_colors(5)
    assert len(set(cols)) == 5
    assert all(max(c) >= 200 for c in cols)
    assert point_colors(5) == cols            # reproducible


def test_render_breaks_line_at_occlusion():
    frames = np.zeros((6, 50, 100, 3), np.uint8)
    tracks = np.array([[10, 25], [20, 25], [30, 25], [70, 25], [80, 25], [90, 25]], np.float32)[:, None]
    vis = np.array([1, 1, 0, 0, 1, 1], np.float32)[:, None]
    last = list(render_frames(frames, tracks, vis, point_radius=1, line_thickness=1))[-1]
    row = last[25].max(axis=-1)
    assert row[15] > 0 and row[85] > 0        # both segments drawn
    assert row[50] == 0                        # no line across the gap (20 -> 80)


# ---------------------------------------------------------------- metrics
def test_metrics_without_gt():
    tracks = np.stack([np.arange(10.0), np.zeros(10)], -1)[:, None]     # 1 px/frame, straight
    m = evaluate_tracking(tracks, np.ones((10, 1)))
    assert m["Smoothness"] == pytest.approx(0.0)
    assert m["Continuity"] == pytest.approx(1.0)
    assert m["Trajectory Length"] == pytest.approx(9.0)


def test_metrics_skip_occluded_steps():
    tracks = np.stack([np.arange(10.0), np.zeros(10)], -1)[:, None]
    tracks[5, 0] = [500, 500]
    vis = np.ones((10, 1)); vis[5] = 0
    m = evaluate_tracking(tracks, vis)
    assert m["Continuity"] == pytest.approx(0.9)
    assert m["Trajectory Length"] == pytest.approx(7.0)   # steps 4->5 and 5->6 excluded


def test_metrics_with_gt():
    gt = np.zeros((5, 2, 2))
    pred = gt + np.array([3.0, 4.0])                      # 5 px error everywhere
    m = evaluate_tracking(pred, np.ones((5, 2)), gt_tracks=gt, gt_visibility=np.ones((5, 2)))
    assert m["EPE"] == pytest.approx(5.0)
    assert m["PCK@5px"] == pytest.approx(0.0)
    assert m["Visibility Accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------- pipeline
def test_make_queries_scales_and_validates():
    q = make_queries([(100, 50), (3, 10, 20)], query_frame=7, scale=0.5, num_frames=10)
    np.testing.assert_allclose(q, [[7, 50, 25], [3, 5, 10]])
    with pytest.raises(ValueError):
        make_queries([(100, 50)], query_frame=10, num_frames=10)
    with pytest.raises(ValueError):
        make_queries([])


def test_track_video_end_to_end(make_video, tmp_path):
    out, csv_path = str(tmp_path / "out" / "tracked.mp4"), str(tmp_path / "out" / "tracks.csv")
    r = track_video(make_video(width=800, height=450, fps=24.0), [(100, 225)], query_frame=0,
                    output_path=out, csv_path=csv_path, max_side=400, tracker_fn=fake_tracker)
    assert r.video.scale == pytest.approx(0.5)
    np.testing.assert_allclose(r.queries, [[0, 50, 112.5]])

    cap = cv2.VideoCapture(out)
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 40
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(24.0, abs=0.01)
    cap.release()

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    assert len(rows) == 40
    assert float(rows[0]["x"]) == pytest.approx(100.0, abs=0.01)    # back in original pixels
    assert rows[12]["visible"] == "0" and rows[20]["visible"] == "1"
    assert r.metrics["Continuity"] == pytest.approx(35 / 40)


def test_cli_reports_bad_video(tmp_path, capsys):
    assert cli_main(["--video", str(tmp_path / "missing.mp4"), "--point", "1,2"]) == 1
    assert "not found" in capsys.readouterr().err


def test_cli_rejects_bad_point():
    with pytest.raises(SystemExit):
        cli_main(["--video", "x.mp4", "--point", "1,2,3,4"])


# ---------------------------------------------------------------- model (no download: fake online model)
class _FakeOnlineModel:
    """Mimics CoTracker's online predictor: each call returns tracks for all frames seen so far."""
    step = 8

    def __call__(self, video_chunk, is_first_step=False, queries=None):
        import torch
        if is_first_step:
            self.n, self.seen = queries.shape[1], 0
            return None, None
        self.seen = self.seen + self.step if self.seen else video_chunk.shape[1]
        t = torch.arange(self.seen, dtype=torch.float32)
        tracks = t[None, :, None, None].expand(1, self.seen, self.n, 2).clone()
        return tracks, torch.ones(1, self.seen, self.n, dtype=torch.bool)


@pytest.mark.parametrize("T", [1, 5, 8, 16, 17, 100])
def test_online_mode_covers_every_frame(T):
    pytest.importorskip("torch")
    from tracker.model import run_cotracker
    frames = np.zeros((T, 32, 48, 3), np.uint8)
    tracks, vis = run_cotracker(frames, np.array([[0, 5, 5]], np.float32), model=_FakeOnlineModel(),
                                device="cpu", mode="online")
    assert tracks.shape == (T, 1, 2) and vis.shape == (T, 1)
    np.testing.assert_allclose(tracks[:, 0, 0], np.arange(T))


def test_resolve_mode():
    pytest.importorskip("torch")
    from tracker.model import resolve_mode
    assert resolve_mode("auto", "cpu") == "online"
    assert resolve_mode("auto", "cuda") == "offline"
    assert resolve_mode("offline", "cpu") == "offline"
    with pytest.raises(ValueError):
        resolve_mode("fast", "cpu")


def test_parse_model_resolution():
    pytest.importorskip("torch")
    from tracker.model import parse_resolution
    assert parse_resolution("192x256") == (192, 256)
    assert parse_resolution((256, 352)) == (256, 352)
    assert parse_resolution(None) is None and parse_resolution("") is None
    for bad in ("200x256", "32x32", "abc"):
        with pytest.raises(ValueError):
            parse_resolution(bad)


def test_model_resolution_is_applied():
    pytest.importorskip("torch")
    from tracker.model import run_cotracker
    m = _FakeOnlineModel()
    m.interp_shape = (384, 512)
    run_cotracker(np.zeros((10, 32, 48, 3), np.uint8), np.array([[0, 5, 5]], np.float32),
                  model=m, device="cpu", mode="online", model_resolution="192x256")
    assert m.interp_shape == (192, 256)

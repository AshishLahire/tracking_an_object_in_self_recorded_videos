"""Web app: upload a video, click on the object, get the tracked video and measurements.

    python app.py                # http://127.0.0.1:7860

Settings (environment variables, useful for hosted demos on CPU):
    MAX_SECONDS   longest part of a video that is processed (default 20)
    MAX_SIDE      frames are downscaled to this longer side (default 640)
"""

import glob
import os
from functools import lru_cache
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import cv2  # noqa: E402
import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tracker import track_video  # noqa: E402

MAX_SECONDS = float(os.environ.get("MAX_SECONDS", 20))
MAX_SIDE = int(os.environ.get("MAX_SIDE", 640))
PREVIEW_SIDE = 720

CLICK_POINT = "Point on the object"
CLICK_BOX = "Box around the object (2 clicks)"
CLICK_REF = "Reference length (2 clicks)"
CLICK_LINE = "Counting line (2 clicks)"
CLICK_ZONE = "Zone corner (3+ clicks)"

EMPTY_SEL = {"points": [], "box": [], "ref": [], "lines": [], "pending_line": [], "zone": []}


# ---------------------------------------------------------------------------- helpers
@lru_cache(maxsize=64)
def _read_frame_cached(path, index):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.setflags(write=False)          # shared between calls: never modify in place
    return frame


def read_frame(path, index):
    frame = _read_frame_cached(path, int(index))
    if frame is None:
        raise gr.Error(f"Could not read frame {index}")
    return frame


def video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise gr.Error("Could not open this video. Try an .mp4 file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # The container's frame count is sometimes a few frames too high: find the last readable one
    for _ in range(30):
        if n <= 1:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
        ok, _frame = cap.read()
        if ok:
            break
        n -= 1
    cap.release()
    return fps, n, w, h


def preview_scale(w, h):
    return min(1.0, PREVIEW_SIDE / max(w, h))


def draw_selection(frame, sel, scale):
    """Draw the current selection (original-pixel coordinates) on a preview frame."""
    img = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
    p = lambda xy: (int(round(xy[0] * scale)), int(round(xy[1] * scale)))  # noqa: E731
    th = 2

    if len(sel["zone"]) >= 2:
        pts = np.array([p(z) for z in sel["zone"]], np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], len(sel["zone"]) >= 3, (0, 160, 255), th, cv2.LINE_AA)
    for z in sel["zone"]:
        cv2.circle(img, p(z), 4, (0, 160, 255), -1)

    for x1, y1, x2, y2 in sel["lines"]:
        cv2.line(img, p((x1, y1)), p((x2, y2)), (255, 215, 0), th, cv2.LINE_AA)
    for xy in sel["pending_line"]:
        cv2.circle(img, p(xy), 5, (255, 215, 0), -1)

    if len(sel["ref"]) == 2:
        cv2.line(img, p(sel["ref"][0]), p(sel["ref"][1]), (255, 0, 255), th, cv2.LINE_AA)
    for xy in sel["ref"]:
        cv2.circle(img, p(xy), 5, (255, 0, 255), -1)

    if len(sel["box"]) == 2:
        cv2.rectangle(img, p(sel["box"][0]), p(sel["box"][1]), (0, 255, 0), th, cv2.LINE_AA)
    for xy in sel["box"]:
        cv2.circle(img, p(xy), 4, (0, 255, 0), -1)

    for i, xy in enumerate(sel["points"]):
        cv2.circle(img, p(xy), 6, (0, 255, 255), -1)
        cv2.circle(img, p(xy), 6, (0, 0, 0), 1)
        cv2.putText(img, str(i), (p(xy)[0] + 7, p(xy)[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return img


def selection_text(sel):
    parts = []
    if sel["points"]:
        parts.append(f"{len(sel['points'])} point(s)")
    if len(sel["box"]) == 2:
        parts.append("box")
    elif sel["box"]:
        parts.append("box: click the opposite corner")
    if len(sel["ref"]) == 2:
        (x1, y1), (x2, y2) = sel["ref"]
        parts.append(f"reference {np.hypot(x2 - x1, y2 - y1):.0f} px")
    elif sel["ref"]:
        parts.append("reference: click the second end")
    if sel["lines"]:
        parts.append(f"{len(sel['lines'])} counting line(s)")
    if sel["pending_line"]:
        parts.append("line: click the second end")
    if sel["zone"]:
        parts.append(f"zone with {len(sel['zone'])} corner(s)")
    return "Selected: " + (", ".join(parts) if parts else "nothing yet. Click on the object in the frame.")


# ---------------------------------------------------------------------------- events
def on_video(path):
    if not path:
        return None, gr.update(value=0, maximum=1), None, dict(EMPTY_SEL), "Upload a video to start."
    fps, n, w, h = video_info(path)
    info = {"path": path, "fps": fps, "frames": n, "w": w, "h": h, "scale": preview_scale(w, h)}
    frame = read_frame(path, 0)
    sel = {k: [] for k in EMPTY_SEL}
    limit = int(MAX_SECONDS * fps)
    note = f"{w}x{h}, {fps:.1f} FPS, {n} frames ({n / fps:.1f} s)."
    if n > limit:
        note += f" Only the first {MAX_SECONDS:g} s ({limit} frames) will be processed."
    return info, gr.update(value=0, maximum=max(n - 1, 1)), draw_selection(frame, sel, info["scale"]), sel, note


def on_frame(info, index, sel):
    if not info:
        return None
    return draw_selection(read_frame(info["path"], index), sel, info["scale"])


def on_click(info, index, sel, mode, evt: gr.SelectData):
    if not info:
        raise gr.Error("Upload a video first.")
    x, y = evt.index[0] / info["scale"], evt.index[1] / info["scale"]    # -> original pixels
    xy = (round(x, 1), round(y, 1))
    if mode == CLICK_POINT:
        sel["points"].append(xy)
    elif mode == CLICK_BOX:
        sel["box"] = [xy] if len(sel["box"]) != 1 else sel["box"] + [xy]
    elif mode == CLICK_REF:
        sel["ref"] = [xy] if len(sel["ref"]) != 1 else sel["ref"] + [xy]
    elif mode == CLICK_LINE:
        sel["pending_line"].append(xy)
        if len(sel["pending_line"]) == 2:
            (x1, y1), (x2, y2) = sel["pending_line"]
            sel["lines"].append((x1, y1, x2, y2))
            sel["pending_line"] = []
    elif mode == CLICK_ZONE:
        sel["zone"].append(xy)
    return sel, draw_selection(read_frame(info["path"], index), sel, info["scale"]), selection_text(sel)


def on_undo(info, index, sel, mode):
    key = {CLICK_POINT: "points", CLICK_BOX: "box", CLICK_REF: "ref", CLICK_ZONE: "zone"}.get(mode)
    if mode == CLICK_LINE:
        if sel["pending_line"]:
            sel["pending_line"].pop()
        elif sel["lines"]:
            sel["lines"].pop()
    elif key and sel[key]:
        sel[key].pop()
    img = draw_selection(read_frame(info["path"], index), sel, info["scale"]) if info else None
    return sel, img, selection_text(sel)


def on_clear(info, index):
    sel = {k: [] for k in EMPTY_SEL}
    img = draw_selection(read_frame(info["path"], index), sel, info["scale"]) if info else None
    return sel, img, selection_text(sel)


def run(info, index, sel, ref_length, ref_is_object, stabilize, smooth, grid, mode, progress=gr.Progress()):
    if not info:
        raise gr.Error("Upload a video first.")
    has_box = len(sel["box"]) == 2
    if not sel["points"] and not has_box:
        raise gr.Error("Click at least one point on the object, or draw a box around it.")

    reference = None
    if len(sel["ref"]) == 2:
        if not ref_length or ref_length <= 0:
            raise gr.Error("Enter the real length of the reference (in metres).")
        (x1, y1), (x2, y2) = sel["ref"]
        reference = (x1, y1, x2, y2, float(ref_length))
    if ref_is_object and reference is not None and not has_box:
        raise gr.Error("'Reference is the object itself' needs a box around the object.")

    max_frames = int(MAX_SECONDS * info["fps"])
    if index >= max_frames:
        raise gr.Error(f"Pick a frame within the first {MAX_SECONDS:g} s of the video.")

    out_dir = tempfile.mkdtemp(prefix="tracker_")
    name = os.path.splitext(os.path.basename(info["path"]))[0].replace(" ", "_")
    box = None
    if has_box:
        (x1, y1), (x2, y2) = sel["box"]
        box = (x1, y1, x2, y2)

    try:
        r = track_video(
            info["path"], sel["points"], query_frame=int(index), box=box, box_grid=int(grid),
            output_path=os.path.join(out_dir, f"{name}_tracked.mp4"),
            csv_path=os.path.join(out_dir, f"{name}_points.csv"),
            json_path=os.path.join(out_dir, f"{name}_summary.json"),
            heatmap_path=os.path.join(out_dir, f"{name}_heatmap.png"),
            max_side=MAX_SIDE, max_frames=max_frames, smooth=smooth, mode=mode,
            reference=reference, reference_is_object=bool(ref_is_object and reference is not None),
            count_lines=sel["lines"] or None,
            zones=[sel["zone"]] if len(sel["zone"]) >= 3 else None,
            stabilize=stabilize, browser_compatible=True,
            progress=lambda f, msg: progress(f, desc=msg))
    except ValueError as e:
        raise gr.Error(str(e))

    m = r.motion
    lines = [
        f"### Results for `{os.path.basename(info['path'])}`",
        f"- **Tracked:** {len(r.queries)} point(s) over {r.video.num_frames} frames "
        f"({r.video.num_frames / r.video.fps:.1f} s), visible {r.metrics['Continuity']:.0%} of the time",
        f"- **Distance:** {m['distance']:.2f} {m['unit_distance']}",
        f"- **Average speed:** {m['avg_speed']:.2f} {m['unit_speed']}  |  **Top speed (95th pct):** "
        f"{m['max_speed']:.2f} {m['unit_speed']}",
    ]
    if reference is None:
        lines.append("- *Add a reference length to get speed in km/h and distance in metres.*")
    if r.stabilized:
        lines.append("- Camera motion compensated (measurements relative to the background)")
    if sel["lines"]:
        lines.append(f"- **Line crossings:** {len(r.crossings)}")
        for c in r.crossings[:20]:
            lines.append(f"  - line {c['line']}: frame {c['frame']} ({c['time_s']} s), {c['direction']}")
    for z in r.zones:
        lines.append(f"- **Zone:** entered {len(z['entries'])} time(s), {z['time_inside_s']} s inside")

    t = np.arange(r.video.num_frames) / r.video.fps
    speed_df = pd.DataFrame({"time (s)": t, f"speed ({m['unit_speed']})": m["speed"]}).dropna()
    files = [r.csv_path, r.object_csv_path, r.json_path, r.heatmap_path]
    return r.output_path, "\n".join(lines), r.heatmap_image, speed_df, files


# ---------------------------------------------------------------------------- layout
INTRO = """
# Object Tracker
Track any object in your own video: upload, click on the object, press **Track**.
Powered by [CoTracker3](https://github.com/facebookresearch/co-tracker).

**How:** 1) upload a video, 2) move the slider to a frame where the object is clearly visible
(the start of its motion works best), 3) click on the object, or switch to *Box* and click two
opposite corners, 4) optionally add a reference length (e.g. a car is about 4.5 m long) for real
speed, a counting line, or a zone, 5) press **Track**.
"""

EXAMPLES = sorted(glob.glob(os.path.join(ROOT, "examples", "*.mp4")))


def build_app():
    with gr.Blocks(title="Object Tracker", theme=gr.themes.Soft()) as demo:
        gr.Markdown(INTRO)
        info = gr.State(None)
        sel = gr.State(dict(EMPTY_SEL))

        with gr.Row():
            with gr.Column(scale=2):
                video_in = gr.Video(label="1. Video", sources=["upload"], height=260)
                if EXAMPLES:
                    gr.Examples(EXAMPLES, inputs=video_in, label="Or try an example")
                note = gr.Markdown("Upload a video to start.")
                frame_idx = gr.Slider(0, 1, value=0, step=1, label="2. Frame to select the object on")
                click_mode = gr.Radio([CLICK_POINT, CLICK_BOX, CLICK_REF, CLICK_LINE, CLICK_ZONE],
                                      value=CLICK_POINT, label="3. What does a click add?")
                ref_len = gr.Number(label="Reference real length (metres)", value=1.0, minimum=0)
                ref_obj = gr.Checkbox(label="The reference is the tracked object itself (e.g. the car's length). "
                                            "Keeps speed right when it moves toward or away from the camera; needs a box.",
                                      value=False)
                with gr.Row():
                    undo_btn = gr.Button("Undo last")
                    clear_btn = gr.Button("Clear all")
                with gr.Accordion("Options", open=False):
                    stabilize = gr.Checkbox(label="Compensate camera shake / panning (handheld videos)", value=False)
                    smooth = gr.Checkbox(label="Smooth trajectories", value=True)
                    grid = gr.Slider(2, 6, value=4, step=1, label="Box mode: points per side")
                    mode = gr.Radio(["auto", "online", "offline"], value="auto",
                                    label="Model mode (auto: offline on GPU, online on CPU)")
                run_btn = gr.Button("Track", variant="primary", size="lg")

            with gr.Column(scale=3):
                preview = gr.Image(label="Click on the frame", type="numpy", interactive=False,
                                   show_download_button=False)
                sel_text = gr.Markdown(selection_text(EMPTY_SEL))

        gr.Markdown("## Results")
        with gr.Row():
            video_out = gr.Video(label="Tracked video")
            summary = gr.Markdown()
        with gr.Row():
            heat = gr.Image(label="Where the object spent time (heatmap)", type="numpy")
            speed_plot = gr.LinePlot(x="time (s)", y="speed", label="Speed over time", height=300)
        files_out = gr.File(label="Downloads (CSV per point, CSV per frame, JSON summary, heatmap)",
                            file_count="multiple")

        video_in.change(on_video, video_in, [info, frame_idx, preview, sel, note]).then(
            lambda s: selection_text(s), sel, sel_text)
        frame_idx.release(on_frame, [info, frame_idx, sel], preview)
        preview.select(on_click, [info, frame_idx, sel, click_mode], [sel, preview, sel_text])
        undo_btn.click(on_undo, [info, frame_idx, sel, click_mode], [sel, preview, sel_text])
        clear_btn.click(on_clear, [info, frame_idx], [sel, preview, sel_text])

        def _run(*args, progress=gr.Progress()):
            vid, md, hm, df, files = run(*args, progress=progress)
            plot = gr.LinePlot(value=df, x=df.columns[0], y=df.columns[1], label="Speed over time", height=300)
            return vid, md, hm, plot, files

        run_btn.click(_run, [info, frame_idx, sel, ref_len, ref_obj, stabilize, smooth, grid, mode],
                      [video_out, summary, heat, speed_plot, files_out])
    return demo


if __name__ == "__main__":
    build_app().queue(max_size=8).launch(server_name=os.environ.get("HOST", "127.0.0.1"),
                                         server_port=int(os.environ.get("PORT", 7860)))

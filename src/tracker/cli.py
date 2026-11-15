"""Command-line interface.

Examples::

    # one point on the cyclist, selected on frame 70
    python cli.py --video examples/bicycle.mp4 --frame 70 --point 125,50

    # box around the car, real speed from a 4.5 m car length, count crossings of a line
    python cli.py --video examples/car_side.mp4 --box 120,100,260,170 \\
        --reference 120,165,260,165,4.5 --line 213,60,213,230 --csv results/car.csv --json results/car.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _numbers(text: str, counts, what: str):
    try:
        values = [float(p) for p in text.replace(" ", "").split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"{what}: expected numbers separated by commas, got {text!r}")
    if len(values) not in counts:
        raise argparse.ArgumentTypeError(f"{what}: expected {' or '.join(map(str, counts))} numbers, got {text!r}")
    return tuple(values)


def _parse_point(text):
    return _numbers(text, (2, 3), "point must be 'x,y' or 'frame,x,y'")


def _parse_box(text):
    return _numbers(text, (4,), "box must be 'x1,y1,x2,y2'")


def _parse_line(text):
    return _numbers(text, (4,), "line must be 'x1,y1,x2,y2'")


def _parse_reference(text):
    return _numbers(text, (5,), "reference must be 'x1,y1,x2,y2,length_m'")


def _parse_joint(text):
    return _numbers(text, (6,), "joint must be 'x1,y1,x2,y2,x3,y3' (end, joint, end)")


def _parse_zone(text):
    try:
        pts = [tuple(float(v) for v in p.split(",")) for p in text.replace(" ", "").split(";") if p]
    except ValueError:
        raise argparse.ArgumentTypeError(f"zone must be 'x,y;x,y;x,y;...', got {text!r}")
    if len(pts) < 3 or any(len(p) != 2 for p in pts):
        raise argparse.ArgumentTypeError(f"zone needs at least 3 'x,y' points separated by ';', got {text!r}")
    return pts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="object-tracker",
        description="Track an object through a video with CoTracker3 and measure its motion. "
                    "All coordinates are pixels of the ORIGINAL video, on --frame.")
    p.add_argument("--video", required=True, help="input video file")

    sel = p.add_argument_group("what to track (give --point and/or --box)")
    sel.add_argument("--point", action="append", type=_parse_point, default=[],
                     help="point on the object, 'x,y' (on --frame) or 'frame,x,y'; repeatable")
    sel.add_argument("--box", type=_parse_box, help="box around the object 'x1,y1,x2,y2'; a grid of points "
                                                    "inside it is tracked and the box follows the object")
    sel.add_argument("--grid", type=int, default=4, help="box mode: grid size (default 4 -> 16 points)")
    sel.add_argument("--joint", action="append", type=_parse_joint, default=[],
                     help="joint angle from 3 points 'x1,y1,x2,y2,x3,y3': one end, the joint, the other end "
                          "(e.g. hip,knee,ankle); reports range of motion and repetitions; repeatable")
    sel.add_argument("--joint-name", action="append", default=[], help="name for each --joint (e.g. knee)")
    sel.add_argument("--frame", type=int, default=0, help="frame on which points / box / lines are given (default 0)")

    mes = p.add_argument_group("measurements")
    mes.add_argument("--reference", type=_parse_reference,
                     help="'x1,y1,x2,y2,length_m': a segment of known real length -> speed in km/h, distance in m")
    mes.add_argument("--reference-is-object", action="store_true",
                     help="the reference was measured on the tracked object (needs --box); keeps speed "
                          "correct when the object moves toward / away from the camera")
    mes.add_argument("--line", action="append", type=_parse_line, default=[],
                     help="counting line 'x1,y1,x2,y2'; reports each crossing; repeatable")
    mes.add_argument("--zone", action="append", type=_parse_zone, default=[],
                     help="zone polygon 'x,y;x,y;x,y;...'; reports entries/exits/time inside; repeatable")
    mes.add_argument("--stabilize", action="store_true",
                     help="compensate camera shake / panning before measuring (handheld videos)")

    out = p.add_argument_group("outputs")
    out.add_argument("--output", help="annotated video (default: results/<name>_tracked.mp4)")
    out.add_argument("--csv", help="per-point CSV; also writes <name>_object.csv with one row per frame")
    out.add_argument("--json", help="JSON summary (video info, speed, crossings, zones, metrics)")
    out.add_argument("--heatmap", help="PNG heatmap of where the object spent time")
    out.add_argument("--browser", action="store_true", help="re-encode the video to H.264 (plays in browsers)")

    run = p.add_argument_group("processing")
    run.add_argument("--max-side", type=int, default=640,
                     help="downscale frames whose longer side is larger (0 = keep original size)")
    run.add_argument("--max-frames", type=int, help="only process the first N frames")
    run.add_argument("--no-smooth", action="store_true", help="use the raw model output")
    run.add_argument("--device", choices=["cpu", "cuda"], help="default: cuda if available")
    run.add_argument("--model-res", help="resolution the model works at, e.g. 192x256 "
                                          "(default 384x512; lower = less memory, faster, slightly less precise)")
    run.add_argument("--mode", choices=["auto", "offline", "online"], default="auto",
                     help="offline = whole video at once (best, needs lots of memory); online = sliding "
                          "windows (low memory, tracks forward only); auto = offline on GPU, online on CPU")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.point and not args.box and not args.joint:
        parser.error("give at least one --point, a --box or a --joint")

    from .metrics import format_metrics
    from .pipeline import track_video

    output = args.output
    if output is None:
        name = os.path.splitext(os.path.basename(args.video))[0].replace(" ", "_")
        output = os.path.join("results", f"{name}_tracked.mp4")

    last = {"step": -1}

    def progress(frac, msg):
        step = int(frac * 20)              # print at most every 5%
        if step != last["step"]:
            last["step"] = step
            print(f"[{frac:4.0%}] {msg}", flush=True)

    try:
        result = track_video(
            args.video, args.point, query_frame=args.frame, box=args.box, box_grid=args.grid,
            output_path=output, csv_path=args.csv, json_path=args.json, heatmap_path=args.heatmap,
            max_side=args.max_side or None, max_frames=args.max_frames,
            smooth=not args.no_smooth, device=args.device, mode=args.mode, model_resolution=args.model_res,
            reference=args.reference, reference_is_object=args.reference_is_object,
            count_lines=args.line, zones=args.zone,
            stabilize=args.stabilize, browser_compatible=args.browser, progress=progress,
            joints=[[j[0:2], j[2:4], j[4:6]] for j in args.joint] or None, joint_names=args.joint_name or None)
    except (FileNotFoundError, IOError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    v = result.video
    m = result.motion
    print(f"\nVideo   : {args.video} ({v.num_frames} frames, {v.fps:.2f} FPS, "
          f"{v.original_size[0]}x{v.original_size[1]} -> {v.size[0]}x{v.size[1]})")
    for label, path in [("Output", result.output_path), ("CSV", result.csv_path),
                        ("Object", result.object_csv_path), ("Joints", result.joints_csv_path),
                        ("JSON", result.json_path),
                        ("Heatmap", result.heatmap_path)]:
        if path:
            print(f"{label:8s}: {path}")
    print(f"Motion  : distance {m['distance']:.1f} {m['unit_distance']}, "
          f"avg {m['avg_speed']:.1f} {m['unit_speed']}, max {m['max_speed']:.1f} {m['unit_speed']}"
          + (" (camera-compensated)" if result.stabilized else ""))
    if args.line:
        print(f"Crossings: {json.dumps(result.crossings)}")
    for z in result.zones:
        print(f"Zone {z['zone']}  : {len(z['entries'])} entries, {z['time_inside_s']} s inside")
    for j in result.joints:
        print(f"{j['name']:8s}: range of motion {j['rom_deg']:.1f} deg ({j['min_deg']:.1f} to {j['max_deg']:.1f}), "
              f"{j['reps']} repetition(s)" + (f", {j['mean_rep_s']:.2f} s each" if j['reps'] >= 2 else ""))
    print(format_metrics(result.metrics, "Metrics (smoothed tracks)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
track_cbiou_offline.py
----------------------
Re-track an HOI video JSON (from demo_video.py) with Roboflow C-BIoU.

Run this in the dedicated `cbiou` conda env (Python >= 3.10), NOT in `codetr`.

Workflow for side-by-side comparison:
  1) codetr:  python demo/demo_video.py          # HybridSORT (or TRACKER=None)
  2) cbiou:   python demo/track_cbiou_offline.py # rewrite track_ids with C-BIoU
  3) either:  python demo/vis_offline.py         # re-render from the new JSON
              (vis_offline still needs packages available in that env, or
               re-run it from codetr pointing at the C-BIoU JSON)

Edit settings below, then:
    conda activate cbiou
    cd /path/to/Hoi-Detr_Hybrid_Sort
    python demo/track_cbiou_offline.py
"""

from __future__ import annotations

import json
import os
import sys

from tqdm import tqdm

# demo/ on path for hoi_trackers + predictions_io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predictions_io import det_to_jsonable, load_detections
from hoi_trackers import build_tracker


# ══════════════════════════════════════════════════════════════
# USER SETTINGS
# ══════════════════════════════════════════════════════════════
# HOI video JSON exported by demo_video.py (detections already present).
INPUT_JSON = "Output/media0.json"

# Where to write the C-BIoU-tracked JSON. None -> <stem>_cbiou.json next to input.
OUTPUT_JSON = None

# C-BIoU knobs (paper: b1 < b2)

CBIOU_BUFFER_RATIO_FIRST = 0.3   # b1 (small buffer, first pass)
CBIOU_BUFFER_RATIO_SECOND = 0.5  # b2 (larger buffer, second pass)
CBIOU_LOST_TRACK_BUFFER = 30
CBIOU_FRAME_RATE = 30.0
CBIOU_TRACK_ACTIVATION_THR = 0.3
CBIOU_MIN_CONSECUTIVE_FRAMES = 1
CBIOU_IOU_THR_FIRST = 0.2
CBIOU_IOU_THR_SECOND = 0.5
CBIOU_IOU_THR_UNCONFIRMED = 0.3
CBIOU_HIGH_CONF_DET_THR = 0.3


def _default_out_path(in_path: str) -> str:
    root, ext = os.path.splitext(in_path)
    return f"{root}_cbiou{ext or '.json'}"


def retrack_video_json(meta: dict, tracker) -> dict:
    """Replace per-frame track_id fields using C-BIoU; keep boxes/interactions."""
    tracker.reset()
    frames_out = []
    for rec in tqdm(meta.get("frames", []), desc="cbiou"):
        dets = load_detections(rec.get("detections", []))
        tracker.update(dets, frame=None)
        frames_out.append({
            **rec,
            "detections": [det_to_jsonable(d) for d in dets],
        })

    out = dict(meta)
    out["frames"] = frames_out
    out["tracker"] = tracker.name
    out["tracker_backend"] = "cbiou_offline"
    out["cbiou"] = {
        "buffer_ratio_first": CBIOU_BUFFER_RATIO_FIRST,
        "buffer_ratio_second": CBIOU_BUFFER_RATIO_SECOND,
        "lost_track_buffer": CBIOU_LOST_TRACK_BUFFER,
        "frame_rate": CBIOU_FRAME_RATE,
        "track_activation_threshold": CBIOU_TRACK_ACTIVATION_THR,
        "minimum_consecutive_frames": CBIOU_MIN_CONSECUTIVE_FRAMES,
        "high_conf_det_threshold": CBIOU_HIGH_CONF_DET_THR,
    }
    return out


def main():
    in_path = INPUT_JSON
    if not os.path.isfile(in_path):
        print(f"[ERROR] INPUT_JSON not found: {in_path}")
        print("Run demo_video.py in `codetr` first (EXPORT_JSON=True).")
        sys.exit(1)

    with open(in_path, "r") as f:
        meta = json.load(f)

    if meta.get("type") != "video":
        print(f"[ERROR] Expected video JSON (type='video'), got {meta.get('type')!r}")
        sys.exit(1)

    tracker = build_tracker(
        "cbiou",
        buffer_ratio_first=CBIOU_BUFFER_RATIO_FIRST,
        buffer_ratio_second=CBIOU_BUFFER_RATIO_SECOND,
        lost_track_buffer=CBIOU_LOST_TRACK_BUFFER,
        frame_rate=float(meta.get("fps") or CBIOU_FRAME_RATE),
        track_activation_threshold=CBIOU_TRACK_ACTIVATION_THR,
        minimum_consecutive_frames=CBIOU_MIN_CONSECUTIVE_FRAMES,
        minimum_iou_threshold_first_assoc=CBIOU_IOU_THR_FIRST,
        minimum_iou_threshold_second_assoc=CBIOU_IOU_THR_SECOND,
        minimum_iou_threshold_unconfirmed_assoc=CBIOU_IOU_THR_UNCONFIRMED,
        high_conf_det_threshold=CBIOU_HIGH_CONF_DET_THR,
    )
    print(f"[INFO] Tracker: {tracker}")
    print(f"[INFO] Frames: {len(meta.get('frames', []))}")

    out_meta = retrack_video_json(meta, tracker)
    out_path = OUTPUT_JSON or _default_out_path(in_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_meta, f)
    print(f"[INFO] Wrote {out_path}")
    print("[INFO] Re-render with codetr:")
    print(f"       # set PREDICTIONS_JSON = '{out_path}' in demo/vis_offline.py")
    print("       conda activate codetr && python demo/vis_offline.py")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Runner for isolated engine v2 (pickup re-grab + putback occlusion-aware).

Does NOT modify standalone_single_camera_tracker.py.

Usage:
  python run_tracker_putback_v2.py \\
      --input media4.mp4 \\
      --output out/26f0_v2.mp4 \\
      --hoi-json path/to_cam1.json \\
      --roi roi/cam1_roi.json --camera-id 1 --conf 0.4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import standalone_single_camera_tracker as base
from putback_engine_v2 import (
    PickupEngineV2,
    PutbackDetectorV2,
    filter_product_detections,
    filter_roi_detections,
)

logger = base.logger


def process_video_v2(
    input_video_path: str,
    output_video_path: str,
    model_path: str = "models/best_10_6.pt",
    conf_threshold: float = 0.5,
    camera_id: int = 0,
    show_preview: bool = False,
    roi_config_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    hoi_json_path: Optional[str] = None,
    write_video: bool = True,
) -> Dict[str, object]:
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video file not found: {input_video_path}")

    output_dir = os.path.dirname(os.path.abspath(output_video_path))
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"[engine-v2] Input: {input_video_path}")
    logger.info(f"[engine-v2] {width}x{height} @ {fps:.2f} FPS | frames={total_frames}")

    writer = None
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not initialize VideoWriter for: {output_video_path}")

    roi_polygon = None
    safe_roi_polygon = None
    if roi_config_path and os.path.exists(roi_config_path):
        try:
            with open(roi_config_path, "r", encoding="utf-8") as f:
                roi_data = json.load(f)
            if "points" in roi_data:
                roi_polygon = np.array(roi_data["points"], dtype=np.float32)
            if "safe_polygon" in roi_data:
                safe_roi_polygon = np.array(roi_data["safe_polygon"], dtype=np.float32)
            logger.info(f"Loaded ROI from {roi_config_path}")
        except Exception as e:
            logger.warning(f"Error loading ROI config: {e}")

    if hoi_json_path:
        detector = base.HOIJsonDetector(
            hoi_json_path=hoi_json_path,
            conf_threshold=conf_threshold,
            compute_cnn_embedding=logger.isEnabledFor(logging.DEBUG),
        )
    else:
        detector = base.YOLODetector(model_path=model_path, conf_threshold=conf_threshold)

    tracker = base.SingleCameraTracker(camera_id=camera_id)
    pickup_engine = PickupEngineV2() if hoi_json_path else None
    putback_detector = PutbackDetectorV2() if hoi_json_path else None

    frame_index = 0
    start_time = time.time()
    # Hand boxes accepted last frame — used so ROI-born hands may leave ROI
    # (carry / open) without spawning background-person tracks.
    prev_hand_boxes: list = []

    if show_preview:
        cv2.namedWindow("engine-v2", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_index += 1
            timestamp_ms = (frame_index / fps) * 1000.0

            detections = detector.detect(
                frame,
                camera_id=camera_id,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
            )
            # Drop whole-machine / door-strip firstobject boxes before tracking.
            detections = filter_product_detections(detections)

            # Products always ROI-gated. Hands birth only inside ROI; out-of-ROI
            # hands only if they continue a shop hand from the previous frame
            # (blocks background people + upper-door grabbers that never entered).
            detections = filter_roi_detections(
                detections, roi_polygon, prev_hand_boxes
            )

            observations = tracker.update(
                detections,
                safe_roi_polygon=safe_roi_polygon,
                full_roi_polygon=roi_polygon,
            )
            prev_hand_boxes = [
                tuple(float(v) for v in o.bbox)
                for o in observations
                if o.class_name == "hand"
            ]

            if pickup_engine is not None and putback_detector is not None:
                frame_hoi = (
                    detector.get_frame_data(frame_index)
                    if hasattr(detector, "get_frame_data") else None
                )
                pickup_engine.process_frame(frame_hoi, observations, frame_index)
                putback_detector.sync_pickups(pickup_engine.pickups)
                putback_detector.process_frame(
                    frame_hoi,
                    observations,
                    frame_index,
                    hand_carrying=pickup_engine.hand_carrying,
                )
                # Latches put-phase so shelf contact during place isn't a new pickup.
                pickup_engine.sync_putbacks(putback_detector.putbacks)

            if write_video and writer is not None:
                elapsed = time.time() - start_time
                current_fps = frame_index / elapsed if elapsed > 0 else 0.0
                active = list(tracker.tracks.values())
                annotated = base.draw_tracking_overlay(
                    frame=frame,
                    active_tracks=active,
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    fps_estimate=current_fps,
                    roi_polygon=roi_polygon,
                    safe_roi_polygon=safe_roi_polygon,
                    pickups_count=(len(pickup_engine.pickups) if pickup_engine else None),
                    putbacks_count=(len(putback_detector.putbacks) if putback_detector else None),
                )
                # banner from whichever event is more recent
                banner = None
                bframe = -10 ** 9
                if pickup_engine and pickup_engine.last_event_text:
                    banner, bframe = pickup_engine.last_event_text, pickup_engine.last_event_frame
                if putback_detector and putback_detector.last_event_frame >= bframe:
                    banner, bframe = putback_detector.last_event_text, putback_detector.last_event_frame
                if banner and (frame_index - bframe) <= base.EVENT_BANNER_PERSIST_FRAMES:
                    annotated = base.draw_event_banner(annotated, banner)
                writer.write(annotated)

                if show_preview:
                    cv2.imshow("engine-v2", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if max_frames and frame_index >= max_frames:
                break
            if frame_index % 100 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"[engine-v2] {frame_index}/{total_frames} "
                    f"({frame_index / elapsed if elapsed else 0:.1f} FPS)"
                )
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()

    total_time = time.time() - start_time
    result: Dict[str, object] = {
        "frames_processed": frame_index,
        "total_processing_time_s": total_time,
        "avg_fps": frame_index / total_time if total_time > 0 else 0.0,
        "output_video_path": output_video_path if write_video else None,
        "engine": "v2_pickup_regrab_putback_occlusion",
    }

    if pickup_engine is not None and putback_detector is not None:
        pickups = pickup_engine.pickups
        putbacks = putback_detector.putbacks
        logger.info(
            f"[engine-v2] Pickups Confirmed: {len(pickups)} | Putbacks Confirmed: {len(putbacks)}"
        )
        result["pickups"] = pickups
        result["putbacks"] = putbacks
        events_path = str(Path(output_video_path).with_suffix("")) + "_pickup_putback.json"
        with open(events_path, "w", encoding="utf-8") as f:
            json.dump({"engine": "v2", "pickups": pickups, "putbacks": putbacks}, f, indent=2)
        logger.info(f"[engine-v2] Events: {events_path}")

    return result


def parse_args():
    p = argparse.ArgumentParser(description="Isolated engine-v2 runner (team file untouched).")
    p.add_argument("--input", "-i", type=str, required=True)
    p.add_argument("--output", "-o", type=str, default="outputs/engine_v2_output.mp4")
    p.add_argument("--model", "-m", type=str, default="models/best_10_6.pt")
    p.add_argument("--hoi-json", type=str, default=None)
    p.add_argument("--conf", "-c", type=float, default=0.5)
    p.add_argument("--camera-id", type=int, default=0)
    p.add_argument("--roi", type=str, default=None)
    p.add_argument("--show", "-s", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--no-video", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    # Official ROIs under roi/cam{N}_roi.json (not cam*_roi_adjusted.json).
    roi_path = args.roi
    if not roi_path:
        candidate = _HERE / "roi" / f"cam{args.camera_id}_roi.json"
        if candidate.is_file():
            roi_path = str(candidate)
            logger.info(f"[engine-v2] Using default ROI: {roi_path}")
    process_video_v2(
        input_video_path=args.input,
        output_video_path=args.output,
        model_path=args.model,
        conf_threshold=args.conf,
        camera_id=args.camera_id,
        show_preview=args.show,
        roi_config_path=roi_path,
        max_frames=args.max_frames,
        hoi_json_path=args.hoi_json,
        write_video=not args.no_video,
    )

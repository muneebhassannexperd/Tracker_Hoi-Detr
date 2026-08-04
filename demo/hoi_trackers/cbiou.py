"""
C-BIoU tracker adapter (Roboflow `trackers.CBIoUTracker`).

Converts HOI-DETR detection dicts <-> supervision.Detections and writes
`track_id` onto each matched detection — same contract as HybridSortTracker.

Requires Python >= 3.10 and:
    pip install git+https://github.com/roboflow/trackers.git

Only CBIoUTracker is used (SORT / ByteTrack / OC-SORT / BoT-SORT are not).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import BaseTracker


def _require_cbiou():
    """Lazy import so HybridSORT users are not forced onto Python 3.10."""
    try:
        import supervision as sv
        from trackers import CBIoUTracker
    except ImportError as exc:
        raise ImportError(
            "CBIoU tracker requires the Roboflow `trackers` package and "
            "Python >= 3.10 (your HOI-DETR `codetr` env is typically 3.7).\n"
            "  pip install git+https://github.com/roboflow/trackers.git\n"
            f"Original error: {exc}"
        ) from exc
    return sv, CBIoUTracker


def hoi_dets_to_supervision(dets: List[dict]):
    """
    Build supervision.Detections from HOI-DETR detection dicts.

    Uses the same fields HybridSORT already consumes: box (xyxy), score,
    class_id. Embeddings are ignored (C-BIoU is box-only).
    """
    sv, _ = _require_cbiou()
    if not dets:
        return sv.Detections.empty()

    xyxy = np.asarray([d["box"][:4] for d in dets], dtype=np.float32)
    confidence = np.asarray([float(d["score"]) for d in dets], dtype=np.float32)
    class_id = np.asarray([int(d["class_id"]) for d in dets], dtype=np.int32)
    return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)


def _box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = map(float, a[:4])
    bx1, by1, bx2, by2 = map(float, b[:4])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def supervision_ids_to_hoi(dets: List[dict], tracked) -> None:
    """
    Write tracker_id from supervision.Detections back onto HOI det dicts.

    Prefer index alignment when lengths match; otherwise greedy IoU match.
    tracker_id < 0 (unmatched low-conf) → track_id=None.
    """
    for d in dets:
        d["track_id"] = None

    if tracked is None or len(tracked) == 0 or not dets:
        return

    tracker_ids = getattr(tracked, "tracker_id", None)
    if tracker_ids is None:
        return

    # Fast path: same count / order as input (typical for trackers.update).
    if len(tracked) == len(dets):
        for i, tid in enumerate(tracker_ids):
            tid_i = int(tid)
            dets[i]["track_id"] = tid_i if tid_i >= 0 else None
        return

    # Fallback: greedy IoU assignment.
    used = set()
    pairs = []
    for di, d in enumerate(dets):
        for ti in range(len(tracked)):
            iou = _box_iou(d["box"], tracked.xyxy[ti])
            if iou > 0:
                pairs.append((iou, di, ti))
    pairs.sort(reverse=True)
    for _, di, ti in pairs:
        if di in used or ti in used:
            continue
        tid_i = int(tracker_ids[ti])
        dets[di]["track_id"] = tid_i if tid_i >= 0 else None
        used.add(di)
        used.add(ti)


class CBIoUSortTracker(BaseTracker):
    """
    Per-class C-BIoU wrapper matching HybridSortTracker's BaseTracker API.

    One CBIoUTracker per HOI class so hand / firstobject / secondobject IDs
    never cross (same policy as HybridSORT). Local C-BIoU IDs are remapped to
    a video-global namespace so hand#0 and firstobject#0 do not collide in UI.
    """

    def __init__(
        self,
        buffer_ratio_first: float = 0.1,
        buffer_ratio_second: float = 0.3,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.3,
        minimum_consecutive_frames: int = 1,
        minimum_iou_threshold_first_assoc: float = 0.2,
        minimum_iou_threshold_second_assoc: float = 0.5,
        minimum_iou_threshold_unconfirmed_assoc: float = 0.3,
        high_conf_det_threshold: float = 0.3,
        instant_first_frame_activation: bool = True,
        class_ids=(0, 1, 2),
        # Accept / ignore HybridSORT kwargs so build_tracker(**shared) is safe.
        **_ignored,
    ):
        if buffer_ratio_first < 0 or buffer_ratio_second < 0:
            raise ValueError("buffer_ratio_* must be >= 0")
        self.buffer_ratio_first = float(buffer_ratio_first)
        self.buffer_ratio_second = float(buffer_ratio_second)
        self.lost_track_buffer = int(lost_track_buffer)
        self.frame_rate = float(frame_rate)
        self.track_activation_threshold = float(track_activation_threshold)
        self.minimum_consecutive_frames = int(minimum_consecutive_frames)
        self.minimum_iou_threshold_first_assoc = float(
            minimum_iou_threshold_first_assoc
        )
        self.minimum_iou_threshold_second_assoc = float(
            minimum_iou_threshold_second_assoc
        )
        self.minimum_iou_threshold_unconfirmed_assoc = float(
            minimum_iou_threshold_unconfirmed_assoc
        )
        self.high_conf_det_threshold = float(high_conf_det_threshold)
        self.instant_first_frame_activation = bool(instant_first_frame_activation)
        self.class_ids = tuple(class_ids)
        self.name = "cbiou"
        self._trackers: Dict[int, object] = {}
        self._local_to_global: Dict[Tuple[int, int], int] = {}
        self._next_global_id = 1
        self.reset()

    def _make_tracker(self):
        _, CBIoUTracker = _require_cbiou()
        return CBIoUTracker(
            buffer_ratio_first=self.buffer_ratio_first,
            buffer_ratio_second=self.buffer_ratio_second,
            lost_track_buffer=self.lost_track_buffer,
            frame_rate=self.frame_rate,
            track_activation_threshold=self.track_activation_threshold,
            minimum_consecutive_frames=self.minimum_consecutive_frames,
            minimum_iou_threshold_first_assoc=self.minimum_iou_threshold_first_assoc,
            minimum_iou_threshold_second_assoc=self.minimum_iou_threshold_second_assoc,
            minimum_iou_threshold_unconfirmed_assoc=(
                self.minimum_iou_threshold_unconfirmed_assoc
            ),
            high_conf_det_threshold=self.high_conf_det_threshold,
            instant_first_frame_activation=self.instant_first_frame_activation,
        )

    def reset(self) -> None:
        self._trackers = {cid: self._make_tracker() for cid in self.class_ids}
        self._local_to_global = {}
        self._next_global_id = 1

    def _to_global(self, class_id: int, local_id: Optional[int]) -> Optional[int]:
        if local_id is None or int(local_id) < 0:
            return None
        key = (int(class_id), int(local_id))
        if key not in self._local_to_global:
            self._local_to_global[key] = self._next_global_id
            self._next_global_id += 1
        return self._local_to_global[key]

    def update(self, detections: List[dict], frame) -> List[dict]:
        # frame is unused by C-BIoU (passing it warns) — omit intentionally.
        for d in detections:
            d["track_id"] = None

        if not detections:
            empty = hoi_dets_to_supervision([])
            for trk in self._trackers.values():
                trk.update(empty)
            return detections

        for cid in self.class_ids:
            idxs = [i for i, d in enumerate(detections) if int(d["class_id"]) == cid]
            cls_dets = [detections[i] for i in idxs]
            sv_dets = hoi_dets_to_supervision(cls_dets)
            tracked = self._trackers[cid].update(sv_dets)
            supervision_ids_to_hoi(cls_dets, tracked)
            for local_i, global_i in enumerate(idxs):
                local_tid = cls_dets[local_i].get("track_id")
                detections[global_i]["track_id"] = self._to_global(cid, local_tid)

        return detections

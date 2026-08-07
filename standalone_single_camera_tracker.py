#!/usr/bin/env python3
"""
Standalone Single Camera Tracker
================================
A self-contained script for running single-camera object tracking on input videos.
This script embeds all tracking algorithms, configurations, geometric utility functions,
and visual rendering routines into a single independent file for testing and deployment.

Usage Example:
    python standalone_single_camera_tracker.py \
        --input videos/24289f08-cc38-4886-bd91-d04f35442f75/media0.mp4 \
        --output outputs/tracking_output_media0.mp4 \
        --model models/best_10_6.pt \
        --conf 0.5 \
        --show
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# Optional Torch / Ultralytics dependencies
try:
    import torch
except ImportError:
    torch = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("StandaloneTracker")


# =============================================================================
# 1. EMBEDDED CONFIGURATION & FAMILY MAPPING
# =============================================================================

# Identity protection & tracking parameters
LOCAL_TRACK_MAX_MISSES: int = 45
LOCAL_TRACK_MAX_DISTANCE_NORM: float = 2.5
LOCAL_TRACK_MIN_SIMILARITY: float = 0.45
# Used only as the reject bar for a match with no box overlap at moderate
# distance (see _cost()) -- found empirically too low at 0.45: a real
# identity-switch case (two different physical PRODUCTS, ~280px apart, IoU
# ~0.02) measured similarity=0.467, just barely above 0.45, so it slipped
# through. Split by class, not a single constant: raising this for hands too
# broke legitimate fast hand motion (a real continuing hand at frame101 had
# similarity=0.582 and IoU=0.0 -- hands move fast enough that low frame-to-
# frame box overlap is normal, unlike a resting/slowly-carried product) --
# hands keep the original, more lenient bar.
LOCAL_TRACK_STRONG_SIMILARITY_FIRSTOBJECT: float = 0.6
LOCAL_TRACK_STRONG_SIMILARITY_HAND: float = 0.45
TEMPORAL_IOU_THRESHOLD: float = 0.1
TRACK_EMBEDDING_BANK: int = 3

# Local tracker family-aware class tolerance settings
LOCAL_TRACK_FAMILY_TOLERANCE_ENABLED: bool = True
LOCAL_TRACK_FAMILY_TOLERANCE_MIN_IOU: float = 0.3
LOCAL_TRACK_FAMILY_TOLERANCE_MAX_DISTANCE_NORM: float = 0.3
LOCAL_TRACK_FAMILY_DISAGREEMENT_PENALTY: float = 0.4
LOCAL_TRACK_REASSIGN_DISTANCE_GAP_PX: float = 150.0

ROI_STABLE_EDGE_MARGIN_PX: float = 140
UNKNOWN_UNSTABLE_CLASS: str = "UNKNOWN_UNSTABLE_CLASS"

# Model detection names mapping
MODEL_TO_DASHBOARD_MAPPING: Dict[str, str] = {
    "Aquafina": "Aquafina",
    "Barebells Caramel Cashew": "Barebells - Caramel Cashew",
    "Barebells Cookies and Cream": "Barebells - Cookies & Cream",
    "Legendary Tasty Pastry - Strawberry": "Legendary Tasty Pastry - Strawberry",
    "One Bar - Hersheys Cookies N Cream": "One Bar - Hershey's Cookies N Cream",
    "One Bar - Reess PB Lovers": "One Bar - Reese's PB Lovers",
    "Quest Chips Chili Lime": "Quest Chips Chili Lime",
    "Quest Chips Hot and Spicy": "Quest Chips - Hot & Spicy",
    "Quest Chips Nacho Cheese": "Quest Chips - Nacho Cheese",
    "Skullcandy DIME 3": "Skullcandy DIME 3",
}


def family_base_class(class_name: str) -> str:
    """Return the coarse product family for a SKU class name."""
    if not class_name:
        return UNKNOWN_UNSTABLE_CLASS
    lower = class_name.lower()
    if lower.startswith("barebells"):
        return "Barebells"
    if lower.startswith("quest chips"):
        return "Quest Chips"
    if lower.startswith("one bar"):
        return "One Bar"
    if lower.startswith("legendary tasty pastry"):
        return "Legendary Tasty Pastry"
    if lower.startswith("aquafina"):
        return "Aquafina"
    if lower.startswith("skullcandy"):
        return "Skullcandy DIME 3"
    return class_name


# =============================================================================
# 2. GEOMETRIC & EMBEDDING UTILITIES
# =============================================================================

def bbox_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0.0 else 0.0


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two feature vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Check if point (x, y) lies inside or on the boundary of polygon."""
    if polygon is None or len(polygon) < 3:
        return True
    pt = (float(point[0]), float(point[1]))
    poly = polygon.astype(np.float32)
    res = cv2.pointPolygonTest(poly, pt, measureDist=False)
    return res >= 0.0


def signed_distance_to_polygon(point: np.ndarray, polygon: np.ndarray) -> float:
    """Compute signed distance from point to polygon (positive inside, negative outside)."""
    if polygon is None or len(polygon) < 3:
        return 100.0
    pt = (float(point[0]), float(point[1]))
    poly = polygon.astype(np.float32)
    return float(cv2.pointPolygonTest(poly, pt, measureDist=True))


def crop_histogram_embedding(frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """Extract a 96-dimensional BGR color histogram embedding from bounding box crop."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], max(x1 + 1, x2))
    y2 = min(frame.shape[0], max(y1 + 1, y2))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(96, dtype=np.float32)

    hist = cv2.calcHist([crop], [0, 1, 2], None, [4, 4, 6], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, None).flatten().astype(np.float32)
    return hist


# =============================================================================
# 3. DATA CLASSES
# =============================================================================

@dataclass
class Detection:
    """Represents an object detection in a single frame."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float
    embedding: np.ndarray
    camera_id: int = 0
    frame_index: int = 0
    timestamp_ms: float = 0.0
    centroid: np.ndarray = field(init=False)
    source_view: str = "original"
    display_bbox: Optional[Tuple[float, float, float, float]] = None
    original_centroid: Optional[np.ndarray] = None
    in_safe_roi_override: Optional[bool] = None
    in_outer_roi_override: Optional[bool] = None

    def __post_init__(self) -> None:
        self.centroid = np.array(
            [(self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0],
            dtype=np.float32
        )


@dataclass
class LocalTrack:
    """A short-term local track for a single camera view."""
    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]
    centroid: np.ndarray
    last_confidence: float
    last_embedding: np.ndarray
    last_frame_index: int
    last_timestamp_ms: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    missed_frames: int = 0
    age: int = 1
    embedding_history: deque = field(default_factory=lambda: deque(maxlen=TRACK_EMBEDDING_BANK))
    prev_bbox: Optional[Tuple[float, float, float, float]] = None
    temporal_iou: float = 1.0
    in_safe_roi_override: Optional[bool] = None
    in_outer_roi_override: Optional[bool] = None
    source_view: str = "original"
    display_bbox: Optional[Tuple[float, float, float, float]] = None
    motion_centroid: Optional[np.ndarray] = None
    motion_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    history: List[Tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.history.append((float(self.centroid[0]), float(self.centroid[1])))

    def update(self, detection: Detection) -> None:
        """Update track with new detection."""
        self.prev_bbox = self.bbox
        self.bbox = detection.bbox
        self.velocity = detection.centroid - self.centroid

        next_motion_centroid = (
            detection.original_centroid.copy() if detection.original_centroid is not None
            else detection.centroid.copy()
        )
        if self.motion_centroid is not None:
            self.motion_velocity = next_motion_centroid - self.motion_centroid

        self.centroid = detection.centroid
        self.history.append((float(self.centroid[0]), float(self.centroid[1])))
        if len(self.history) > 30:
            self.history.pop(0)

        self.last_confidence = detection.confidence
        self.last_embedding = detection.embedding
        self.last_frame_index = detection.frame_index
        self.last_timestamp_ms = detection.timestamp_ms
        self.missed_frames = 0
        self.age += 1
        self.embedding_history.append(detection.embedding)

        self.temporal_iou = 1.0 if self.prev_bbox is None else bbox_iou(self.prev_bbox, self.bbox)
        self.in_safe_roi_override = detection.in_safe_roi_override
        self.in_outer_roi_override = detection.in_outer_roi_override
        self.source_view = detection.source_view
        self.display_bbox = detection.display_bbox
        self.motion_centroid = next_motion_centroid


@dataclass
class TrackObservation:
    """Standardized output observation for a track at a specific frame."""
    camera_id: int
    local_track_id: int
    frame_index: int
    timestamp_ms: float
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]
    centroid: np.ndarray
    confidence: float
    embedding: np.ndarray
    velocity: np.ndarray
    motion_centroid: np.ndarray
    display_bbox: Optional[Tuple[float, float, float, float]]
    in_safe_roi: bool
    in_outer_roi: bool
    in_stable_roi: bool
    safe_roi_distance: float
    temporal_iou: float


# =============================================================================
# 4. SINGLE CAMERA TRACKER ENGINE
# =============================================================================

class SingleCameraTracker:
    """
    Object Tracker for single camera view.
    Matches detections to existing tracks via Hungarian algorithm,
    creates new tracks, and purges stale tracks after max missed frames.
    """

    def __init__(self, camera_id: int = 0) -> None:
        self.camera_id = camera_id
        self.next_track_id = itertools.count(1)
        self.tracks: Dict[int, LocalTrack] = {}

    def _cost(
        self,
        track: LocalTrack,
        detection: Detection,
        safe_roi_polygon: Optional[np.ndarray] = None,
    ) -> float:
        """Calculate matching cost between existing track and detection (lower is better)."""
        centroid_distance = np.linalg.norm(track.centroid - detection.centroid) / 200.0
        iou = bbox_iou(track.bbox, detection.bbox)

        class_penalty = 0.0
        if track.class_id != detection.class_id:
            same_family = False
            if LOCAL_TRACK_FAMILY_TOLERANCE_ENABLED:
                track_family = family_base_class(track.class_name)
                det_family = family_base_class(detection.class_name)
                same_family = (
                    track_family == det_family
                    and bool(track_family)
                    and track_family != UNKNOWN_UNSTABLE_CLASS
                )
            if not same_family:
                return 1e6

            spatially_continuous = (
                iou >= LOCAL_TRACK_FAMILY_TOLERANCE_MIN_IOU
                or centroid_distance <= LOCAL_TRACK_FAMILY_TOLERANCE_MAX_DISTANCE_NORM
            )
            if not spatially_continuous:
                return 1e6
            class_penalty = LOCAL_TRACK_FAMILY_DISAGREEMENT_PENALTY

        if safe_roi_polygon is not None:
            track_distance = signed_distance_to_polygon(track.centroid, safe_roi_polygon)
            if track_distance < 0.0:
                detection_distance = signed_distance_to_polygon(detection.centroid, safe_roi_polygon)
                if detection_distance - track_distance > LOCAL_TRACK_REASSIGN_DISTANCE_GAP_PX:
                    return 1e6

        similarity = cosine_similarity(track.last_embedding, detection.embedding)

        # Reject if genuinely far regardless of anything else, OR if there's
        # no box overlap at all AND appearance isn't a strong match -- the
        # old version required distance to ALSO cross the far cutoff before
        # rejecting on top of low IoU/similarity, which meant a moderate
        # (~250-300px) jump between two different physical objects could
        # never be rejected purely on shape+appearance evidence, no matter
        # how bad both were, as long as it stayed under the distance cutoff.
        no_spatial_overlap = iou < TEMPORAL_IOU_THRESHOLD
        too_far = centroid_distance > LOCAL_TRACK_MAX_DISTANCE_NORM
        strong_similarity_bar = (
            LOCAL_TRACK_STRONG_SIMILARITY_HAND if track.class_name == "hand"
            else LOCAL_TRACK_STRONG_SIMILARITY_FIRSTOBJECT
        )
        appearance_not_strong = similarity < strong_similarity_bar

        if too_far or (no_spatial_overlap and appearance_not_strong):
            return 1e6

        age_bias = min(track.age / 100.0, 0.2)
        return 0.45 * (1.0 - similarity) + 0.35 * centroid_distance + 0.2 * (1.0 - iou) - age_bias + class_penalty

    def _make_observation(
        self,
        track: LocalTrack,
        safe_roi_polygon: Optional[np.ndarray],
        full_roi_polygon: Optional[np.ndarray],
    ) -> TrackObservation:
        """Create TrackObservation from LocalTrack."""
        safe_dist = (
            signed_distance_to_polygon(track.centroid, safe_roi_polygon)
            if safe_roi_polygon is not None else 100.0
        )
        in_safe_roi = (
            track.in_safe_roi_override
            if track.in_safe_roi_override is not None
            else safe_dist >= 0.0
        )
        in_outer_roi = (
            track.in_outer_roi_override
            if track.in_outer_roi_override is not None
            else point_in_polygon(track.centroid, full_roi_polygon)
        )
        in_stable_roi = bool(
            in_safe_roi
            or (in_outer_roi and safe_dist >= -float(ROI_STABLE_EDGE_MARGIN_PX))
        )

        return TrackObservation(
            camera_id=self.camera_id,
            local_track_id=track.track_id,
            frame_index=track.last_frame_index,
            timestamp_ms=track.last_timestamp_ms,
            class_id=track.class_id,
            class_name=track.class_name,
            bbox=track.bbox,
            centroid=track.centroid.copy(),
            confidence=track.last_confidence,
            embedding=track.last_embedding.copy(),
            velocity=track.velocity.copy(),
            motion_centroid=(
                track.motion_centroid.copy() if track.motion_centroid is not None
                else track.centroid.copy()
            ),
            display_bbox=track.display_bbox,
            in_safe_roi=in_safe_roi,
            in_outer_roi=in_outer_roi,
            in_stable_roi=in_stable_roi,
            safe_roi_distance=float(safe_dist),
            temporal_iou=track.temporal_iou,
        )

    def update(
        self,
        detections: List[Detection],
        safe_roi_polygon: Optional[np.ndarray] = None,
        full_roi_polygon: Optional[np.ndarray] = None,
    ) -> List[TrackObservation]:
        """Update tracker with frame detections."""
        track_ids = list(self.tracks.keys())

        if track_ids and detections:
            cost_matrix = np.full((len(track_ids), len(detections)), 1e6, dtype=np.float32)
            for r, track_id in enumerate(track_ids):
                for c, detection in enumerate(detections):
                    cost_matrix[r, c] = self._cost(self.tracks[track_id], detection, safe_roi_polygon)

            rows, cols = linear_sum_assignment(cost_matrix)

            assigned_tracks = set()
            assigned_detections = set()
            for r, c in zip(rows, cols):
                if cost_matrix[r, c] >= 1e5:
                    continue
                track = self.tracks[track_ids[r]]
                track.update(detections[c])
                assigned_tracks.add(track.track_id)
                assigned_detections.add(c)
        else:
            assigned_tracks = set()
            assigned_detections = set()

        updated_track_ids = set(assigned_tracks)

        # Create new tracks for unmatched detections
        for idx, detection in enumerate(detections):
            if idx in assigned_detections:
                continue
            track_id = next(self.next_track_id)
            track = LocalTrack(
                track_id=track_id,
                class_id=detection.class_id,
                class_name=detection.class_name,
                bbox=detection.bbox,
                centroid=detection.centroid.copy(),
                last_confidence=detection.confidence,
                last_embedding=detection.embedding.copy(),
                last_frame_index=detection.frame_index,
                last_timestamp_ms=detection.timestamp_ms,
                in_safe_roi_override=detection.in_safe_roi_override,
                in_outer_roi_override=detection.in_outer_roi_override,
                source_view=detection.source_view,
                display_bbox=detection.display_bbox,
                motion_centroid=(
                    detection.original_centroid.copy() if detection.original_centroid is not None
                    else detection.centroid.copy()
                ),
            )
            track.embedding_history.append(detection.embedding.copy())
            self.tracks[track_id] = track
            updated_track_ids.add(track_id)

        # Handle missed frames and stale track removal
        stale_ids = []
        for track_id, track in list(self.tracks.items()):
            if track_id in assigned_tracks:
                continue
            track.missed_frames += 1
            if track.missed_frames > LOCAL_TRACK_MAX_MISSES:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            self.tracks.pop(track_id, None)

        observations = [
            self._make_observation(self.tracks[tid], safe_roi_polygon, full_roi_polygon)
            for tid in self.tracks
            if tid in updated_track_ids
        ]
        return observations


# =============================================================================
# 5. YOLO OBJECT DETECTOR WRAPPER
# =============================================================================

class YOLODetector:
    """Wrapper for Ultralytics YOLO/RT-DETR object detection model."""

    def __init__(self, model_path: str, device: Optional[str] = None, conf_threshold: float = 0.5) -> None:
        self.model_path = model_path
        if device is None:
            self.device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device
        self.conf_threshold = conf_threshold
        self.model = None

        path = Path(model_path)
        if YOLO is not None and path.exists():
            try:
                self.model = YOLO(str(path))
                logger.info(f"Loaded YOLO/RT-DETR model from {path} on device {self.device}")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
        else:
            logger.warning(f"YOLO package or model file not found at {path}. Pipeline running without detector.")

    def detect(
        self,
        frame: np.ndarray,
        camera_id: int = 0,
        frame_index: int = 0,
        timestamp_ms: float = 0.0
    ) -> List[Detection]:
        """Detect objects in a frame and return Detection objects."""
        if self.model is None:
            return []

        results = self.model.predict(
            [frame],
            verbose=False,
            device=self.device,
            conf=self.conf_threshold
        )
        if not results:
            return []

        result = results[0]
        detections: List[Detection] = []
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()

        for bbox, class_id, confidence in zip(xyxy, cls, conf):
            bbox_tuple = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            class_name = names.get(int(class_id), str(class_id))
            embedding = crop_histogram_embedding(frame, bbox_tuple)
            det = Detection(
                bbox=bbox_tuple,
                class_id=int(class_id),
                class_name=class_name,
                confidence=float(confidence),
                embedding=embedding,
                camera_id=camera_id,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                original_centroid=np.array(
                    [(bbox_tuple[0] + bbox_tuple[2]) / 2.0, (bbox_tuple[1] + bbox_tuple[3]) / 2.0],
                    dtype=np.float32
                ),
            )
            detections.append(det)

        return detections


# =============================================================================
# 5b. HOI-DETR JSON DETECTOR (reads precomputed detections instead of
#     running a live model -- swaps in for YOLODetector)
# =============================================================================

# Near-duplicate concurrent detections within a single frame -- measured
# empirically across all 36 real HOI-DETR sessions available. For
# "firstobject", same-frame same-class pairs closer than ~250px are almost
# always one physical item double-detected (a clear valley in the real
# distance histogram between a duplicate cluster peaking at 40-60px and a
# genuinely-distinct-objects cluster starting at 280px+). Hands need a
# DIFFERENT criterion: real distinct hands can legitimately sit as close as
# ~18px apart (two-handed use), so distance would wrongly merge them --
# IoU>0.5 is a near-perfect discriminator instead (only 0.08% of real
# two-hand frames exceed it, and those few are all <=66px apart).
FIRSTOBJECT_DEDUP_DIST_PX: float = 250.0
HAND_DEDUP_IOU: float = 0.5


def dedupe_same_frame_detections(detections: List["Detection"]) -> List["Detection"]:
    """Suppress near-duplicate detections of the same physical object/hand
    within one frame, keeping the highest-confidence one per cluster."""
    by_class: Dict[str, List["Detection"]] = defaultdict(list)
    for d in detections:
        by_class[d.class_name].append(d)

    keep: List["Detection"] = []
    for class_name, dets in by_class.items():
        dets_sorted = sorted(dets, key=lambda d: -d.confidence)
        suppressed = [False] * len(dets_sorted)
        for i in range(len(dets_sorted)):
            if suppressed[i]:
                continue
            keep.append(dets_sorted[i])
            for j in range(i + 1, len(dets_sorted)):
                if suppressed[j]:
                    continue
                if class_name == "hand":
                    is_duplicate = bbox_iou(dets_sorted[i].bbox, dets_sorted[j].bbox) > HAND_DEDUP_IOU
                else:
                    dist = float(np.linalg.norm(dets_sorted[i].centroid - dets_sorted[j].centroid))
                    is_duplicate = dist < FIRSTOBJECT_DEDUP_DIST_PX
                if is_duplicate:
                    suppressed[j] = True
    return keep


class HOIJsonDetector:
    """
    Reads precomputed HOI-DETR detections from a JSON file (the format
    produced by run_hoidetr.py: {"frames": [{"frame_idx", "detections": [...]}]})
    instead of running a live model. Produces the same Detection objects as
    YOLODetector, so it's a drop-in swap for the SingleCameraTracker below --
    only "hand" and "firstobject" classes are kept (matches what's tracked
    elsewhere in this project; "secondobject" is not tracked here).
    """

    def __init__(self, hoi_json_path: str, conf_threshold: float = 0.5) -> None:
        self.conf_threshold = conf_threshold
        with open(hoi_json_path, "r", encoding="utf-8") as f:
            hoi_data = json.load(f)
        self.frames_by_idx: Dict[int, dict] = {
            f["frame_idx"]: f for f in hoi_data.get("frames", [])
        }
        logger.info(
            f"Loaded HOI-DETR JSON from {hoi_json_path} "
            f"({len(self.frames_by_idx)} frames with detections)"
        )

    def detect(
        self,
        frame: np.ndarray,
        camera_id: int = 0,
        frame_index: int = 0,
        timestamp_ms: float = 0.0,
    ) -> List[Detection]:
        """Look up this frame's HOI-DETR detections and build Detection
        objects, computing the embedding from the real frame crop (the JSON
        only stores box/score/class, not an embedding)."""
        # process_video_standalone increments frame_index to 1 for the first
        # frame read; HOI-DETR's own frame_idx is 0-based, so shift by one.
        frame_data = self.frames_by_idx.get(frame_index - 1)
        if frame_data is None:
            return []

        detections: List[Detection] = []
        for det in frame_data.get("detections", []):
            class_name = det.get("class_name")
            if class_name not in ("hand", "firstobject"):
                continue
            score = float(det.get("score", 0.0))
            if score < self.conf_threshold:
                continue

            bbox_tuple = tuple(float(v) for v in det["box"])
            embedding = crop_histogram_embedding(frame, bbox_tuple)
            detections.append(Detection(
                bbox=bbox_tuple,
                class_id=int(det.get("class_id", 0)),
                class_name=class_name,
                confidence=score,
                embedding=embedding,
                camera_id=camera_id,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                original_centroid=np.array(
                    [(bbox_tuple[0] + bbox_tuple[2]) / 2.0, (bbox_tuple[1] + bbox_tuple[3]) / 2.0],
                    dtype=np.float32,
                ),
            ))
        return dedupe_same_frame_detections(detections)

    def get_frame_data(self, frame_index: int) -> Optional[dict]:
        """Raw HOI-DETR frame entry (detections + hf links), for callers that
        need the hf link list directly (the pickup/putback engine below) --
        same frame_index -> frame_idx shift as detect()."""
        return self.frames_by_idx.get(frame_index - 1)


# =============================================================================
# 6. PICKUP/PUTBACK ENGINE (HOI-DETR Episode logic)
#    Pickup and putback are symmetric, standalone counters:
#      - Pickup:  hf link + touch + outward motion (score >= +bar)
#      - Putback: hf link + touch + inward motion  (score <= -bar)
#    No pairing between them -- putback does not cancel a prior pickup.
# =============================================================================

# --- Contact (touch) evidence ---
CONTACT_FLOOR: float = 0.35        # min hf link_conf to count as "touching" this frame
CONTACT_K1: int = 5                # consecutive touching frames required to leave RESTING

# --- Motion evidence (ROI-boundary depth, hand-primary) ---
MOTION_WINDOW: int = 5             # frames back for the hand_delta comparison
BASE_WEIGHT: float = 1.0
COUPLING_BONUS: float = 0.5        # added when hand<->object offset is a stable rigid pair
GESTURE_BONUS: float = 0.5         # added when the linked hand's grip reads CLOSED
COUPLING_OFFSET_STD_PX: float = 25.0  # max std of recent hand-object offsets to count as "coupled"

# --- State transition bars ---
PICKUP_BAR_HIGH: float = 40.0      # required outward_score if claimed almost instantly after contact
PICKUP_BAR_LOW: float = 15.0       # required outward_score once contact has held up for a while
PICKUP_BAR_RELAX_FRAMES: int = 20  # frames-since-contact-start at which the bar fully relaxes to LOW
MIN_TRANSITION_GAP: int = 15       # refractory window (frames) between any two confirmed transitions
NOISE_FLOOR: float = 5.0           # outward_score below this when contact drops to 0 -> false start

# --- Hand physical-state cross-check ---
# A hand that already picked something up shouldn't easily get credited with
# a SECOND, different pickup moments later without stronger evidence -- a
# real hand can't casually hold several loose items while still actively
# reaching for more. Only meaningful now that hand track IDs are stable
# (see LOCAL_TRACK_STRONG_SIMILARITY_HAND) -- before that fix this would
# have just been noise. Grace period, not a permanent lock: once enough time
# has passed since the hand last touched its held item, it's plausible the
# item was stowed off-camera (a bag/basket), so the penalty fades rather
# than blocking the rest of the session.
HAND_BUSY_BAR_MULTIPLIER: float = 1.6
HAND_BUSY_GRACE_FRAMES: int = 90   # ~3s @30fps

# --- Episode garbage collection ---
RESTING_IDLE_DROP_FRAMES: int = 30 # RESTING + zero contact_streak this long -> safe to garbage-collect

# --- Track-reassignment reconciliation ---
# The underlying SingleCameraTracker's own local track ID can break and
# reappear under a new ID for what is really the same physical object/spot
# (occlusion during a grasp, or a near-duplicate detection nearby) -- found
# empirically running against real footage (eeb4886b_cam1: object_track 4
# ending at frame 315, object_track 9 starting at frame 318, ~95px away, same
# shelf slot). Rather than minting a fresh episode (and a spurious extra
# pickup) for the new ID, reconcile it back onto the existing episode when
# it's a close, recent match.
REASSIGN_MAX_GAP_FRAMES: int = 90     # ~3s @30fps -- only reconcile a short gap
REASSIGN_PROXIMITY_PX: float = 120.0  # candidate must be spatially close to the last known position

# --- hf link (raw HOI-DETR box pair) -> live TrackObservation resolution ---
HF_TRACK_MATCH_IOU: float = 0.3

# --- Hand gesture (open/closed) heuristic ---
# No dedicated gesture classifier is wired up yet, so this uses the pseudocode's
# stated fallback: hand-box compactness (aspect ratio) versus the same hand
# track's own recent resting shape. Replace with a real crop classifier later
# if this heuristic proves too noisy once actually run against footage.
GESTURE_BASELINE_LEN: int = 15
GESTURE_COMPACTNESS_DELTA: float = 0.18

GRIP_OPEN = "OPEN"
GRIP_CLOSED = "CLOSED"
GRIP_UNKNOWN = "UNKNOWN"

EP_RESTING = "RESTING"
EP_TRANSITIONING = "TRANSITIONING"


def graduated_bar(since_contact: int) -> float:
    """High bar if pickup is being claimed almost instantly after contact
    started (protects against a hand briefly brushing past an item looking
    like a grab); relaxes down to the low bar as more frames pass, giving
    real evidence more time to accumulate."""
    if since_contact <= 3:
        return PICKUP_BAR_HIGH
    if since_contact >= PICKUP_BAR_RELAX_FRAMES:
        return PICKUP_BAR_LOW
    t = (since_contact - 3) / float(PICKUP_BAR_RELAX_FRAMES - 3)
    return PICKUP_BAR_HIGH - t * (PICKUP_BAR_HIGH - PICKUP_BAR_LOW)


def _value_n_back(history: Sequence[Tuple], n: int, field_index: int):
    """Return a tuple-field value from n steps back in a (frame_idx, ...)
    history sequence, or the oldest available entry if the episode hasn't
    lived long enough yet to have n+1 entries."""
    if not history:
        return None
    idx = -min(n + 1, len(history))
    return history[idx][field_index]


def _entry_n_back(history: Sequence[Tuple], n: int) -> Optional[Tuple]:
    """Same lookup as _value_n_back, but returns the full entry (including
    its real frame_idx at [0]) so the caller can tell how stale it actually
    is -- entries are only appended on TOUCHED frames, so "n steps back" can
    silently span far more than n real frames after a contact gap."""
    if not history:
        return None
    idx = -min(n + 1, len(history))
    return history[idx]


@dataclass
class Episode:
    """Tracks one hand<->object interaction on a single object_track_id.
    Pickup and putback are independent events: either can fire from
    TRANSITIONING when outward_score crosses +/- bar. After a putback,
    putback_done locks further putbacks on this episode/object track;
    pickup still resets and can fire again as before."""
    episode_id: int
    object_track_id: int
    state: str = EP_RESTING
    linked_hand_id: Optional[int] = None
    contact_streak: int = 0
    contact_start_frame: Optional[int] = None
    last_transition_frame: int = 0
    hand_position_history: deque = field(default_factory=lambda: deque(maxlen=MOTION_WINDOW + 1))
    object_position_history: deque = field(default_factory=lambda: deque(maxlen=300))
    grip_history: deque = field(default_factory=lambda: deque(maxlen=30))
    offset_history: deque = field(default_factory=lambda: deque(maxlen=20))
    outward_score: float = 0.0
    frames_since_seen: int = 0
    product_name: Optional[str] = None
    putback_done: bool = False  # at most one putback per object track / episode


class PickupPutbackEngine:
    """Runs the Episode state machine across frames. Consumes this frame's
    raw HOI-DETR hf links (resolved to live track IDs) plus this frame's
    TrackObservation list from SingleCameraTracker.update()."""

    def __init__(self) -> None:
        self.episodes: Dict[int, Episode] = {}   # keyed by object_track_id
        self._episode_id_counter = itertools.count(1)
        self._touched_episode_ids_this_frame: set = set()
        self._hand_aspect_baseline: Dict[int, deque] = {}
        self.pickups: List[dict] = []
        self.putbacks: List[dict] = []
        self.last_event_text: Optional[str] = None
        self.last_event_frame: int = -10 ** 9  # far in the past, so no banner shows before any event
        # hand_track_id -> (episode_id it's carrying, frame it was last touched)
        self.hand_carrying: Dict[int, Tuple[int, int]] = {}
        # Survives episode GC/recreate: same object_track_id never putbacks twice
        self._putback_done_object_ids: Set[int] = set()

    # -- hf link resolution ------------------------------------------------

    @staticmethod
    def _best_iou_match(box: Sequence[float], observations: List["TrackObservation"]) -> Optional["TrackObservation"]:
        box_tuple = tuple(float(v) for v in box)
        best_obs, best_iou = None, HF_TRACK_MATCH_IOU
        for obs in observations:
            iou = bbox_iou(box_tuple, obs.bbox)
            if iou > best_iou:
                best_obs, best_iou = obs, iou
        return best_obs

    def _resolve_hf_links_to_tracks(
        self,
        frame_data: Optional[dict],
        observations: List["TrackObservation"],
    ) -> List[Tuple[int, Optional[int], float]]:
        """Match raw HOI-DETR hf box-pairs (indices into this frame's
        `detections` list) to this frame's live TrackObservation track IDs
        via IoU -- the same box was just fed into the tracker this frame, so
        a correct match has IoU ~1.0."""
        if not frame_data:
            return []
        detections = frame_data.get("detections", [])
        hf = frame_data.get("hf", [])
        if not detections or not hf:
            return []

        hand_obs = [o for o in observations if o.class_name == "hand"]
        object_obs = [o for o in observations if o.class_name == "firstobject"]

        links: List[Tuple[int, Optional[int], float]] = []
        for entry in hf:
            a_idx, b_idx = entry.get("a"), entry.get("b")
            if a_idx is None or b_idx is None or a_idx >= len(detections) or b_idx >= len(detections):
                continue
            link_conf = float(entry.get("prob", 0.0))
            det_a, det_b = detections[a_idx], detections[b_idx]
            if det_a.get("class_name") == "hand" and det_b.get("class_name") == "firstobject":
                hand_det, object_det = det_a, det_b
            elif det_b.get("class_name") == "hand" and det_a.get("class_name") == "firstobject":
                hand_det, object_det = det_b, det_a
            else:
                continue  # not a hand<->firstobject pair (e.g. hand<->secondobject) -- not tracked here

            hand_track = self._best_iou_match(hand_det["box"], hand_obs)
            if hand_track is None:
                continue
            object_track = self._best_iou_match(object_det["box"], object_obs)
            links.append((
                hand_track.local_track_id,
                object_track.local_track_id if object_track is not None else None,
                link_conf,
            ))
        return links

    # -- episode lookup ------------------------------------------------------

    def _reassign_matching_episode(
        self,
        new_track_id: int,
        object_obs: "TrackObservation",
        frame_idx: int,
    ) -> Optional[Episode]:
        """Before minting a brand new episode for an object_track_id never
        seen before, check whether it's really an existing episode's object
        continuing under a reassigned track ID -- a recent, spatially close
        episode that hasn't been touched since -- rather than a genuinely new
        interaction. Skips episodes with no accumulated progress (RESTING,
        zero contact streak): nothing there worth preserving."""
        best_episode = None
        best_dist = None
        best_gap = None
        for episode in self.episodes.values():
            if episode.object_track_id == new_track_id:
                continue
            if episode.state == EP_RESTING and episode.contact_streak == 0:
                continue
            if not episode.object_position_history:
                continue
            gap = episode.frames_since_seen
            if gap > REASSIGN_MAX_GAP_FRAMES:
                continue
            _, last_pos = episode.object_position_history[-1]
            dist = float(np.linalg.norm(np.array(last_pos) - object_obs.centroid))
            if dist > REASSIGN_PROXIMITY_PX:
                continue
            if best_dist is None or dist < best_dist:
                best_episode, best_dist, best_gap = episode, dist, gap

        if best_episode is None:
            return None

        old_track_id = best_episode.object_track_id
        self.episodes.pop(old_track_id, None)
        best_episode.object_track_id = new_track_id
        self.episodes[new_track_id] = best_episode
        logger.info(
            f"[TRACK-REASSIGN] episode={best_episode.episode_id} object_track "
            f"{old_track_id} -> {new_track_id} (frame={frame_idx}, dist={best_dist:.0f}px, gap={best_gap}f)"
        )
        return best_episode

    def _find_or_create_episode(
        self,
        object_track_id: int,
        object_obs: Optional["TrackObservation"],
        frame_idx: int,
    ) -> Episode:
        """Gates purely on 'is this object_track_id already claimed by an
        open episode' -- never on appearance -- which is what keeps
        identical-looking products from being confused with each other. The
        one exception is a track-ID reassignment for the SAME physical
        object/spot (see _reassign_matching_episode), checked only when this
        exact ID has genuinely never been claimed before."""
        episode = self.episodes.get(object_track_id)
        if episode is not None:
            return episode

        if object_obs is not None:
            matched = self._reassign_matching_episode(object_track_id, object_obs, frame_idx)
            if matched is not None:
                return matched

        episode = Episode(episode_id=next(self._episode_id_counter), object_track_id=object_track_id)
        if object_track_id in self._putback_done_object_ids:
            episode.putback_done = True
        self.episodes[object_track_id] = episode
        return episode

    # -- evidence accumulation ------------------------------------------------

    def _update_contact_signal(self, episode: Episode, link_conf: float, frame_idx: int) -> None:
        if link_conf >= CONTACT_FLOOR:
            episode.contact_streak += 1
        else:
            episode.contact_streak = 0

        if episode.state == EP_RESTING and episode.contact_streak >= CONTACT_K1:
            episode.state = EP_TRANSITIONING
            episode.contact_start_frame = frame_idx

    def _offset_is_stable(self, episode: Episode) -> bool:
        if len(episode.offset_history) < 3:
            return False
        offsets = np.array(episode.offset_history, dtype=np.float32)
        return bool(np.linalg.norm(offsets.std(axis=0)) <= COUPLING_OFFSET_STD_PX)

    def _update_motion_evidence(
        self,
        episode: Episode,
        hand_obs: "TrackObservation",
        object_obs: Optional["TrackObservation"],
        frame_idx: int,
    ) -> None:
        # signed_distance_past_roi: safe_roi_distance is positive INSIDE the
        # safe/shelf polygon (machine side, at rest) and negative outside it
        # (customer side) -- so its negation is exactly the "positive =
        # outward/customer side" depth the pseudocode's ROI-line model calls for.
        depth_now = -hand_obs.safe_roi_distance
        hand_pos = (float(hand_obs.centroid[0]), float(hand_obs.centroid[1]))
        episode.hand_position_history.append((frame_idx, hand_pos, depth_now))

        prev_entry = _entry_n_back(episode.hand_position_history, MOTION_WINDOW)
        depth_prev = prev_entry[2]
        hand_delta = depth_now - depth_prev

        # Entries only get appended on TOUCHED frames, so "MOTION_WINDOW
        # steps back" can silently reference a point from many more REAL
        # frames ago after a contact gap -- discount the delta in proportion,
        # since a given distance covered over a long silent gap is much
        # weaker evidence of a real, continuous motion than the same
        # distance covered over the intended few-frame window (found
        # empirically: a hand resting/oscillating near the shelf for ~10
        # frames, then one touch 17 frames later, produced a small but
        # real-looking "recent" delta purely from comparing against a stale
        # 22-frame-old reference point -- enough to falsely cross the bar).
        real_frame_gap = frame_idx - prev_entry[0]
        if real_frame_gap > MOTION_WINDOW:
            hand_delta *= MOTION_WINDOW / float(real_frame_gap)

        weight = BASE_WEIGHT
        if object_obs is not None:
            episode.object_position_history.append(
                (frame_idx, (float(object_obs.centroid[0]), float(object_obs.centroid[1])))
            )
            episode.offset_history.append(hand_obs.centroid - object_obs.centroid)
            if self._offset_is_stable(episode):
                weight += COUPLING_BONUS

        if episode.grip_history and episode.grip_history[-1][1] == GRIP_CLOSED:
            weight += GESTURE_BONUS

        episode.outward_score += hand_delta * weight  # distance past the ROI, not a frame count

    def _estimate_gesture(self, hand_obs: "TrackObservation") -> str:
        x1, y1, x2, y2 = hand_obs.bbox
        w, h = x2 - x1, y2 - y1
        if h <= 1e-6:
            return GRIP_UNKNOWN
        aspect = w / h
        baseline = self._hand_aspect_baseline.setdefault(
            hand_obs.local_track_id, deque(maxlen=GESTURE_BASELINE_LEN)
        )
        reading = GRIP_UNKNOWN
        if len(baseline) >= 5:
            baseline_mean = float(np.mean(baseline))
            if baseline_mean > 1e-6:
                ratio = aspect / baseline_mean
                if ratio <= 1.0 - GESTURE_COMPACTNESS_DELTA:
                    reading = GRIP_CLOSED
                elif ratio >= 1.0 + GESTURE_COMPACTNESS_DELTA:
                    reading = GRIP_OPEN
        baseline.append(aspect)
        return reading

    def _update_gesture_evidence(self, episode: Episode, hand_obs: "TrackObservation", frame_idx: int) -> None:
        episode.grip_history.append((frame_idx, self._estimate_gesture(hand_obs)))

    # -- state transitions ------------------------------------------------

    def _reset_episode_after_event(self, episode: Episode, frame_idx: int) -> None:
        """Return episode to RESTING so the same track can fire again later."""
        episode.state = EP_RESTING
        episode.outward_score = 0.0
        episode.contact_streak = 0
        episode.contact_start_frame = None
        episode.last_transition_frame = frame_idx

    def _evaluate_transition(self, episode: Episode, frame_idx: int) -> None:
        if episode.contact_start_frame is None:
            return
        since_contact = frame_idx - episode.contact_start_frame
        required_bar = graduated_bar(since_contact)
        enough_gap = (frame_idx - episode.last_transition_frame) >= MIN_TRANSITION_GAP

        if episode.state == EP_TRANSITIONING:
            # cross-check against the hand's own physical state: if this same
            # hand is already carrying a DIFFERENT episode (and touched it
            # recently), a second simultaneous pickup needs stronger evidence
            busy = self.hand_carrying.get(episode.linked_hand_id)
            if busy is not None and busy[0] != episode.episode_id:
                busy_episode_id, busy_since_frame = busy
                if (frame_idx - busy_since_frame) <= HAND_BUSY_GRACE_FRAMES:
                    required_bar *= HAND_BUSY_BAR_MULTIPLIER

            if episode.outward_score >= required_bar and enough_gap:
                self._confirm_pickup(episode, frame_idx)
            elif episode.outward_score <= -required_bar and enough_gap:
                if episode.putback_done or episode.object_track_id in self._putback_done_object_ids:
                    logger.info(
                        f"[PUTBACK-IGNORED] frame={frame_idx} episode={episode.episode_id} "
                        f"object_track={episode.object_track_id} already put back once "
                        f"(same episode/object id -- not counting again)"
                    )
                else:
                    self._confirm_putback(episode, frame_idx)
            elif episode.contact_streak == 0 and abs(episode.outward_score) < NOISE_FLOOR:
                episode.state = EP_RESTING
                episode.outward_score = 0.0
                episode.contact_start_frame = None

    # -- confirmation ------------------------------------------------

    def _confirm_pickup(self, episode: Episode, frame_idx: int) -> None:
        self.pickups.append({
            "episode_id": episode.episode_id,
            "frame": frame_idx,
            "hand_track_id": episode.linked_hand_id,
            "object_track_id": episode.object_track_id,
        })
        logger.info(
            f"[PICKUP] episode={episode.episode_id} frame={frame_idx} "
            f"hand_track={episode.linked_hand_id} object_track={episode.object_track_id}"
        )
        self.last_event_text = f"PICKUP CONFIRMED  episode={episode.episode_id}"
        self.last_event_frame = frame_idx
        self.hand_carrying[episode.linked_hand_id] = (episode.episode_id, frame_idx)
        self._reset_episode_after_event(episode, frame_idx)
        # product identity is intentionally NOT resolved here -- left to a
        # later, separate crop-extraction/identification step, per the design

    def _confirm_putback(self, episode: Episode, frame_idx: int) -> None:
        """Standalone putback: mirror of pickup with inward motion. Not paired
        to any prior pickup -- just counts how many putbacks happened.
        At most one putback per object_track_id / episode (locks after first)."""
        episode.putback_done = True
        self._putback_done_object_ids.add(episode.object_track_id)
        self.putbacks.append({
            "episode_id": episode.episode_id,
            "frame": frame_idx,
            "hand_track_id": episode.linked_hand_id,
            "object_track_id": episode.object_track_id,
        })
        logger.info(
            f"[PUTBACK] episode={episode.episode_id} frame={frame_idx} "
            f"hand_track={episode.linked_hand_id} object_track={episode.object_track_id}"
        )
        self.last_event_text = f"PUTBACK CONFIRMED  episode={episode.episode_id}"
        self.last_event_frame = frame_idx
        # Reset score/contact so pickup path stays unchanged; putback stays
        # locked via putback_done / _putback_done_object_ids.
        self._reset_episode_after_event(episode, frame_idx)

    # -- missing evidence / session end ------------------------------------------------

    def _handle_missing_evidence(self, frame_idx: int) -> None:
        for episode in list(self.episodes.values()):
            if episode.episode_id in self._touched_episode_ids_this_frame:
                continue
            # single camera: a gap here just means the hand/product briefly
            # wasn't detected -- freeze, don't decay or reset, expected not an error
            episode.frames_since_seen += 1
            if (episode.state == EP_RESTING and episode.contact_streak == 0
                    and episode.frames_since_seen > RESTING_IDLE_DROP_FRAMES):
                self.episodes.pop(episode.object_track_id, None)  # cheap GC, no info lost

    # -- per-frame entry point ------------------------------------------------

    def process_frame(
        self,
        frame_data: Optional[dict],
        observations: List["TrackObservation"],
        frame_idx: int,
    ) -> None:
        """One call per processed frame. `frame_data` is this frame's raw
        HOI-DETR entry (for its `hf` links, via HOIJsonDetector.get_frame_data);
        `observations` is this frame's TrackObservation list, straight from
        SingleCameraTracker.update()."""
        self._touched_episode_ids_this_frame = set()
        obs_by_track_id = {o.local_track_id: o for o in observations}

        for hand_track_id, object_track_id, link_conf in self._resolve_hf_links_to_tracks(frame_data, observations):
            if object_track_id is None:
                continue
            object_obs = obs_by_track_id.get(object_track_id)
            episode = self._find_or_create_episode(object_track_id, object_obs, frame_idx)
            episode.linked_hand_id = hand_track_id  # always overwritten, never inherited
            episode.frames_since_seen = 0
            self._touched_episode_ids_this_frame.add(episode.episode_id)

            hand_obs = obs_by_track_id.get(hand_track_id)
            if hand_obs is None:
                continue

            self._update_contact_signal(episode, link_conf, frame_idx)
            self._update_motion_evidence(episode, hand_obs, object_obs, frame_idx)
            self._update_gesture_evidence(episode, hand_obs, frame_idx)
            self._evaluate_transition(episode, frame_idx)

        self._handle_missing_evidence(frame_idx)


# =============================================================================
# 7. VISUALIZATION & VIDEO RENDERING
# =============================================================================

def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a distinct, bright BGR color per track ID."""
    np.random.seed(track_id * 17 + 42)
    color = np.random.randint(50, 255, size=3).tolist()
    return (int(color[0]), int(color[1]), int(color[2]))


EVENT_BANNER_PERSIST_FRAMES: int = 45  # ~1.5s @30fps


def draw_event_banner(frame: np.ndarray, event_text: str) -> np.ndarray:
    """Draw a bottom-of-frame banner for a just-confirmed pickup/putback
    event -- the tracking overlay alone (boxes/IDs) doesn't otherwise show
    when an event actually fired."""
    out = frame.copy()
    color = (0, 140, 0) if event_text.startswith("PICKUP") else (0, 0, 200)
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, h - 40), (w, h), color, -1)
    cv2.putText(out, event_text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def draw_tracking_overlay(
    frame: np.ndarray,
    active_tracks: List[LocalTrack],
    frame_index: int,
    timestamp_ms: float,
    fps_estimate: float,
    roi_polygon: Optional[np.ndarray] = None,
    safe_roi_polygon: Optional[np.ndarray] = None,
    pickups_count: Optional[int] = None,
    putbacks_count: Optional[int] = None,
) -> np.ndarray:
    """Draw bounding boxes, track labels, centroids, and motion trajectories on frame."""
    out = frame.copy()

    # Draw ROI polygons if present
    if roi_polygon is not None and len(roi_polygon) >= 3:
        cv2.polylines(out, [roi_polygon.astype(np.int32)], isClosed=True, color=(255, 255, 0), thickness=2)
    if safe_roi_polygon is not None and len(safe_roi_polygon) >= 3:
        cv2.polylines(out, [safe_roi_polygon.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw tracks
    for track in active_tracks:
        color = get_track_color(track.track_id)
        x1, y1, x2, y2 = [int(v) for v in track.bbox]

        # Draw box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=2)

        # Draw trajectory history trail
        if len(track.history) > 1:
            pts = np.array(track.history, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [pts], isClosed=False, color=color, thickness=2)

        # Draw centroid
        cx, cy = int(track.centroid[0]), int(track.centroid[1])
        cv2.circle(out, (cx, cy), radius=4, color=color, thickness=-1)

        # Label tag
        label = f"ID:{track.track_id} {track.class_name} {track.last_confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty1 = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1, ty1), (x1 + tw + 6, ty1 + th + 6), color, -1)
        cv2.putText(out, label, (x1 + 3, ty1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw header info panel
    header = f"Frame: {frame_index} | Time: {timestamp_ms / 1000.0:.2f}s | Active Tracks: {len(active_tracks)} | FPS: {fps_estimate:.1f}"
    if pickups_count is not None:
        header += f" | Pickups: {pickups_count} | Putbacks: {putbacks_count}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 32), (30, 30, 30), -1)
    cv2.putText(out, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return out


# =============================================================================
# 8. MAIN PROCESSING PIPELINE
# =============================================================================

def process_video_standalone(
    input_video_path: str,
    output_video_path: str,
    model_path: str = "models/best_10_6.pt",
    conf_threshold: float = 0.5,
    camera_id: int = 0,
    show_preview: bool = False,
    roi_config_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    hoi_json_path: Optional[str] = None,
) -> Dict[str, object]:
    """
    Process input video frame by frame with object detection & single camera tracker,
    rendering tracking overlays and saving output video.
    """
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video file not found: {input_video_path}")

    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_video_path))
    os.makedirs(output_dir, exist_ok=True)

    # Load video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Input Video: {input_video_path}")
    logger.info(f"Resolution: {width}x{height} @ {fps:.2f} FPS | Total Frames: {total_frames}")

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not initialize VideoWriter for: {output_video_path}")

    # Load ROI polygons if provided
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
            logger.info(f"Loaded ROI configuration from {roi_config_path}")
        except Exception as e:
            logger.warning(f"Error loading ROI config: {e}")

    # Initialize Detector & Tracker
    if hoi_json_path:
        detector = HOIJsonDetector(hoi_json_path=hoi_json_path, conf_threshold=conf_threshold)
    else:
        detector = YOLODetector(model_path=model_path, conf_threshold=conf_threshold)
    tracker = SingleCameraTracker(camera_id=camera_id)

    # The pickup/putback engine needs HOI-DETR's raw hf (hand<->object) link
    # data, which only HOIJsonDetector carries -- so it's only active when a
    # --hoi-json source is in use, not for the live-YOLO path.
    pickup_putback_engine = PickupPutbackEngine() if hoi_json_path else None

    frame_index = 0
    start_time = time.time()
    total_tracks_created = 0

    if show_preview:
        cv2.namedWindow("Standalone Tracker Preview", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_index += 1
            timestamp_ms = (frame_index / fps) * 1000.0

            # 1. Run detection
            detections = detector.detect(
                frame,
                camera_id=camera_id,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms
            )

            # 1b. Restrict PRODUCT detection/tracking to inside the ROI -- the
            # outer ROI polygon was previously only used for cost-gating/
            # labeling downstream, so anything outside it still got tracked
            # and drawn. Hands are deliberately EXEMPT: a real hand moves in
            # and out of the ROI legitimately (idle up high, active inside
            # the shelf), and found empirically that filtering hands the same
            # way silently drops the customer's idle second hand -- removing
            # exactly the context needed to tell two hands apart when the
            # active one switches, since two hands look near-identical by
            # color histogram alone without that positional evidence.
            if roi_polygon is not None:
                detections = [
                    d for d in detections
                    if d.class_name == "hand" or point_in_polygon(d.centroid, roi_polygon)
                ]

            # 2. Update tracker
            observations = tracker.update(
                detections,
                safe_roi_polygon=safe_roi_polygon,
                full_roi_polygon=roi_polygon
            )

            # 2b. Feed this frame's hf links + track observations into the
            # pickup/putback engine (HOI-DETR source only -- see above)
            if pickup_putback_engine is not None:
                frame_hoi_data = (
                    detector.get_frame_data(frame_index)
                    if hasattr(detector, "get_frame_data") else None
                )
                pickup_putback_engine.process_frame(frame_hoi_data, observations, frame_index)

            # Active tracks for rendering
            active_tracks = list(tracker.tracks.values())
            total_tracks_created = tracker.next_track_id.__indirect_self__ if hasattr(tracker.next_track_id, "__indirect_self__") else tracker.next_track_id

            # 3. Draw tracking overlay
            elapsed_sec = time.time() - start_time
            current_fps = frame_index / elapsed_sec if elapsed_sec > 0 else 0.0

            annotated_frame = draw_tracking_overlay(
                frame=frame,
                active_tracks=active_tracks,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                fps_estimate=current_fps,
                roi_polygon=roi_polygon,
                safe_roi_polygon=safe_roi_polygon,
                pickups_count=(len(pickup_putback_engine.pickups) if pickup_putback_engine is not None else None),
                putbacks_count=(len(pickup_putback_engine.putbacks) if pickup_putback_engine is not None else None),
            )

            if (pickup_putback_engine is not None
                    and pickup_putback_engine.last_event_text is not None
                    and (frame_index - pickup_putback_engine.last_event_frame) <= EVENT_BANNER_PERSIST_FRAMES):
                annotated_frame = draw_event_banner(annotated_frame, pickup_putback_engine.last_event_text)

            # 4. Write output frame
            writer.write(annotated_frame)

            # 5. Optional preview
            if show_preview:
                cv2.imshow("Standalone Tracker Preview", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Preview terminated by user ('q' pressed).")
                    break

            if max_frames and frame_index >= max_frames:
                logger.info(f"Reached max frame limit: {max_frames}")
                break

            if frame_index % 100 == 0:
                logger.info(f"Processed {frame_index}/{total_frames} frames ({current_fps:.1f} FPS)...")

    finally:
        cap.release()
        writer.release()
        if show_preview:
            cv2.destroyAllWindows()

    total_processing_time = time.time() - start_time
    avg_fps = frame_index / total_processing_time if total_processing_time > 0 else 0.0

    logger.info("=" * 60)
    logger.info("Tracking Completed!")
    logger.info(f"Processed Frames: {frame_index}")
    logger.info(f"Total Processing Time: {total_processing_time:.2f} s")
    logger.info(f"Average FPS: {avg_fps:.2f}")
    logger.info(f"Output Video Saved: {output_video_path}")

    result: Dict[str, object] = {
        "frames_processed": frame_index,
        "total_processing_time_s": total_processing_time,
        "avg_fps": avg_fps,
        "output_video_path": output_video_path,
    }

    if pickup_putback_engine is not None:
        pickups = pickup_putback_engine.pickups
        putbacks = pickup_putback_engine.putbacks
        logger.info(f"Pickups Confirmed: {len(pickups)} | Putbacks Confirmed: {len(putbacks)}")
        result["pickups"] = pickups
        result["putbacks"] = putbacks

        events_path = str(Path(output_video_path).with_suffix("")) + "_pickup_putback.json"
        with open(events_path, "w", encoding="utf-8") as f:
            json.dump({"pickups": pickups, "putbacks": putbacks}, f, indent=2)
        logger.info(f"Pickup/Putback Events Saved: {events_path}")
        result["events_json_path"] = events_path

    logger.info("=" * 60)

    return result


# =============================================================================
# 9. CLI ENTRYPOINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone Single Camera Tracker for testing video inputs."
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input MP4 video file.")
    parser.add_argument("--output", "-o", type=str, default="outputs/standalone_tracker_output.mp4", help="Path to save output MP4 video file.")
    parser.add_argument("--model", "-m", type=str, default="models/best_10_6.pt", help="Path to YOLO/RT-DETR model weights file.")
    parser.add_argument("--hoi-json", type=str, default=None,
                         help="Path to a precomputed HOI-DETR JSON (from run_hoidetr.py). "
                              "When provided, detections are read from this JSON (hand + "
                              "firstobject classes only) instead of running --model live.")
    parser.add_argument("--conf", "-c", type=float, default=0.5, help="Confidence threshold for detection.")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera identifier (default: 0).")
    parser.add_argument("--roi", type=str, default=None, help="Optional path to ROI config JSON file.")
    parser.add_argument("--show", "-s", action="store_true", help="Display live preview window while processing.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video_standalone(
        input_video_path=args.input,
        output_video_path=args.output,
        model_path=args.model,
        conf_threshold=args.conf,
        camera_id=args.camera_id,
        show_preview=args.show,
        roi_config_path=args.roi,
        max_frames=args.max_frames,
        hoi_json_path=args.hoi_json,
    )

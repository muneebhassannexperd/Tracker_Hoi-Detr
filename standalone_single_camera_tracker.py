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
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

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

        if (iou < TEMPORAL_IOU_THRESHOLD
                and centroid_distance > LOCAL_TRACK_MAX_DISTANCE_NORM
                and similarity < LOCAL_TRACK_MIN_SIMILARITY):
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
# 6. VISUALIZATION & VIDEO RENDERING
# =============================================================================

def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a distinct, bright BGR color per track ID."""
    np.random.seed(track_id * 17 + 42)
    color = np.random.randint(50, 255, size=3).tolist()
    return (int(color[0]), int(color[1]), int(color[2]))


def draw_tracking_overlay(
    frame: np.ndarray,
    active_tracks: List[LocalTrack],
    frame_index: int,
    timestamp_ms: float,
    fps_estimate: float,
    roi_polygon: Optional[np.ndarray] = None,
    safe_roi_polygon: Optional[np.ndarray] = None,
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
    cv2.rectangle(out, (0, 0), (out.shape[1], 32), (30, 30, 30), -1)
    cv2.putText(out, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return out


# =============================================================================
# 7. MAIN PROCESSING PIPELINE
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
    detector = YOLODetector(model_path=model_path, conf_threshold=conf_threshold)
    tracker = SingleCameraTracker(camera_id=camera_id)

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

            # 2. Update tracker
            tracker.update(
                detections,
                safe_roi_polygon=safe_roi_polygon,
                full_roi_polygon=roi_polygon
            )

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
            )

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
    logger.info("=" * 60)

    return {
        "frames_processed": frame_index,
        "total_processing_time_s": total_processing_time,
        "avg_fps": avg_fps,
        "output_video_path": output_video_path
    }


# =============================================================================
# 8. CLI ENTRYPOINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone Single Camera Tracker for testing video inputs."
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input MP4 video file.")
    parser.add_argument("--output", "-o", type=str, default="outputs/standalone_tracker_output.mp4", help="Path to save output MP4 video file.")
    parser.add_argument("--model", "-m", type=str, default="models/best_10_6.pt", help="Path to YOLO/RT-DETR model weights file.")
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
    )

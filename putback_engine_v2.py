"""
engine_v2.py — isolated pickup/putback improvements over the current team tracker.

Does NOT modify standalone_single_camera_tracker.py (team file).

Targets (26f0cc7c cam1, user labels):
  - missing PICKUP around frames 301-315  (same object_track silent gap, then re-grab
    while still marked HELD — engine never re-fires)
  - missing/late PUTBACK 760-790         (object occluded; briefly visible ~770-772
    and ~777-780; in_stable_roi + grip heuristics drop the cycle)

Run via: python run_tracker_putback_v2.py ...
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

import standalone_single_camera_tracker as base
from standalone_single_camera_tracker import (
    EP_HELD,
    EP_RESTING,
    EP_TRANSITIONING,
    Episode,
    GESTURE_BASELINE_LEN,
    GESTURE_COMPACTNESS_DELTA,
    GRIP_CLOSED,
    GRIP_OPEN,
    GRIP_UNKNOWN,
    HAND_BUSY_BAR_MULTIPLIER,
    HAND_BUSY_GRACE_FRAMES,
    HF_TRACK_MATCH_IOU,
    MIN_TRANSITION_GAP,
    NOISE_FLOOR,
    PickupPutbackEngine,
    TrackObservation,
    bbox_iou,
    graduated_bar,
    logger,
)


def _bbox_wh(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def is_machine_scale_bbox(
    box: Tuple[float, float, float, float],
    frame_w: float = 1920.0,
    frame_h: float = 1080.0,
) -> bool:
    """Reject whole-machine / door-strip FO (huge area or ultra-wide + large).

    Aspect alone must NOT reject small product boxes: on 1804478e the real
    product mid-lift at f194 was ~194×60 (aspect≈3.23) and was dropped as a
    'door strip', breaking CONTACT_K1 so the 3rd pickup never left RESTING.
    Ultra-wide aspect only counts when the box is also large (width/area).
    """
    w, h = _bbox_wh(box)
    if w < 1.0 or h < 1.0:
        return True
    area = w * h
    frame_area = max(frame_w * frame_h, 1.0)
    width_frac = w / max(frame_w, 1.0)
    area_frac = area / frame_area
    aspect = w / h
    if area_frac >= MACHINE_FO_MAX_AREA_FRAC:
        return True
    if width_frac >= MACHINE_FO_MAX_WIDTH_FRAC:
        return True
    # Door / cabinet strips are both flat AND large; small flat products are not.
    if (
        aspect >= MACHINE_FO_MAX_ASPECT
        and (width_frac >= 0.28 or area_frac >= 0.025)
    ):
        return True
    return False


def is_product_like_vs_hand(
    hand_box: Tuple[float, float, float, float],
    object_box: Tuple[float, float, float, float],
) -> bool:
    """Hand-relative size geometry (HOI-DETR style): reject oversized FO vs hand."""
    hw, hh = _bbox_wh(hand_box)
    ow, oh = _bbox_wh(object_box)
    if hw < 1.0 or hh < 1.0 or ow < 1.0 or oh < 1.0:
        return False
    hand_area = hw * hh
    obj_area = ow * oh
    violations = 0
    if (ow / hw) > REL_MAX_WIDTH_RATIO:
        violations += 1
    if (oh / hh) > REL_MAX_HEIGHT_RATIO:
        violations += 1
    if (obj_area / hand_area) > (REL_MAX_SCALE_RATIO ** 2):
        violations += 1
    return violations < REL_MIN_VIOLATIONS_TO_REJECT


def is_product_like_observation(
    hand_obs: TrackObservation,
    object_obs: Optional[TrackObservation],
) -> bool:
    if object_obs is None:
        return False
    if object_obs.class_name != "firstobject":
        return False
    if is_machine_scale_bbox(object_obs.bbox):
        return False
    return is_product_like_vs_hand(hand_obs.bbox, object_obs.bbox)


def hand_in_door_band(hand_obs: TrackObservation, frame_h: float = 1080.0) -> bool:
    """Bottom door grip zone (hand low in frame) — not product shelf place."""
    _x1, y1, _x2, y2 = hand_obs.bbox
    cy = 0.5 * (y1 + y2)
    return cy >= frame_h * DOOR_HAND_CY_FRAC or y2 >= frame_h * DOOR_HAND_Y2_FRAC


def filter_product_detections(detections: List) -> List:
    """Drop machine-scale firstobject detections before tracking (v2 runner)."""
    out = []
    for d in detections:
        if getattr(d, "class_name", None) == "firstobject":
            if is_machine_scale_bbox(d.bbox):
                continue
        out.append(d)
    return out


# Hands may leave the machine ROI after a real approach (carry / reach-out),
# but new hands must be born inside ROI so background people never track.
HAND_ROI_CONTINUE_IOU: float = 0.12
# Envelope around outer ROI: continue only if still near the machine.
# Above-top allows multipick lift-out (e7ee ~f150–220, cy well above mouth)
# without tracking faces/bystanders in the full upper frame.
HAND_CONTINUE_ABOVE_TOP_PX: float = 260.0
HAND_CONTINUE_SIDE_PX: float = 60.0
HAND_CONTINUE_BELOW_PX: float = 40.0


def _roi_top_y(roi_polygon: np.ndarray) -> float:
    return float(np.min(roi_polygon[:, 1]))


def _hand_in_continue_envelope(
    cx: float,
    cy: float,
    roi_polygon: np.ndarray,
) -> bool:
    """True if centroid is still near the machine opening (not full-frame free)."""
    xs = roi_polygon[:, 0]
    ys = roi_polygon[:, 1]
    top = float(np.min(ys))
    bot = float(np.max(ys))
    x_lo = float(np.min(xs))
    x_hi = float(np.max(xs))
    return (
        (x_lo - HAND_CONTINUE_SIDE_PX) <= cx <= (x_hi + HAND_CONTINUE_SIDE_PX)
        and (top - HAND_CONTINUE_ABOVE_TOP_PX) <= cy <= (bot + HAND_CONTINUE_BELOW_PX)
    )


def filter_roi_detections(
    detections: List,
    roi_polygon: Optional[np.ndarray],
    prev_hand_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
    prev_product_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
) -> List:
    """ROI gating:

    - firstobject: centroid in outer ROI, OR lift-out band above mouth with
      optional IoU continuity to a prior product (multipick carry).
    - hands: birth only inside outer ROI. Outside only if IoU to a hand kept
      last frame AND still inside a tight envelope around the outer ROI
      (shop lift-out / lateral exit). No free tracking across the whole frame.
    """
    if roi_polygon is None or len(roi_polygon) < 3:
        return list(detections)

    prev = list(prev_hand_boxes or [])
    prev_prod = list(prev_product_boxes or [])
    top_y = _roi_top_y(roi_polygon)
    xs = roi_polygon[:, 0]
    x_lo, x_hi = float(np.min(xs)), float(np.max(xs))
    out = []
    for d in detections:
        name = getattr(d, "class_name", None)
        if name != "hand":
            cx = float(d.centroid[0]) if hasattr(d.centroid, "__len__") else float(d.centroid)
            cy = float(d.centroid[1]) if hasattr(d.centroid, "__len__") else float(d.centroid)
            if base.point_in_polygon(d.centroid, roi_polygon):
                out.append(d)
                continue
            # Narrow lift-out band for products only (not full free frame).
            if (
                (top_y - HAND_CONTINUE_ABOVE_TOP_PX) <= cy <= (top_y + 80.0)
                and x_lo - HAND_CONTINUE_SIDE_PX <= cx <= x_hi + HAND_CONTINUE_SIDE_PX
            ):
                out.append(d)
                continue
            db = tuple(float(v) for v in d.bbox)
            for pb in prev_prod:
                if bbox_iou(db, pb) >= HAND_ROI_CONTINUE_IOU:
                    # Product continue only near envelope too.
                    if _hand_in_continue_envelope(cx, cy, roi_polygon):
                        out.append(d)
                    break
            continue

        # --- hands ---
        cx = float(d.centroid[0]) if hasattr(d.centroid, "__len__") else float(d.centroid)
        cy = float(d.centroid[1]) if hasattr(d.centroid, "__len__") else float(d.centroid)

        if base.point_in_polygon(d.centroid, roi_polygon):
            out.append(d)
            continue

        # Outside polygon: require shop continuity + still near machine envelope.
        if not _hand_in_continue_envelope(cx, cy, roi_polygon):
            continue
        db = tuple(float(v) for v in d.bbox)
        for pb in prev:
            if bbox_iou(db, pb) >= HAND_ROI_CONTINUE_IOU:
                out.append(d)
                break
    return out


def resolve_hf_links_to_tracks(
    frame_data: Optional[dict],
    observations: List[TrackObservation],
) -> List[Tuple[int, Optional[int], float]]:
    """Module-level hf→track matcher (team API made this a private method)."""
    if not frame_data:
        return []
    detections = frame_data.get("detections", [])
    hf = frame_data.get("hf", [])
    if not detections or not hf:
        return []

    hand_obs = [o for o in observations if o.class_name == "hand"]
    object_obs = [
        o for o in observations
        if o.class_name == "firstobject" and not is_machine_scale_bbox(o.bbox)
    ]

    def _best(box, pool):
        box_t = tuple(float(v) for v in box)
        best_obs, best_iou = None, HF_TRACK_MATCH_IOU
        for obs in pool:
            iou = bbox_iou(box_t, obs.bbox)
            if iou > best_iou:
                best_obs, best_iou = obs, iou
        return best_obs

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
            continue
        obj_box = tuple(float(v) for v in object_det["box"])
        hand_box = tuple(float(v) for v in hand_det["box"])
        # Whole-machine / door-strip FO must never drive contact or place.
        if is_machine_scale_bbox(obj_box):
            continue
        if not is_product_like_vs_hand(hand_box, obj_box):
            continue
        hand_track = _best(hand_det["box"], hand_obs)
        if hand_track is None:
            continue
        object_track = _best(object_det["box"], object_obs)
        links.append((
            hand_track.local_track_id,
            object_track.local_track_id if object_track is not None else None,
            link_conf,
        ))
    return links


def estimate_grip(hand_obs: TrackObservation, baseline: deque) -> str:
    """Hand open/closed from aspect ratio vs resting baseline (team heuristic)."""
    x1, y1, x2, y2 = hand_obs.bbox
    w, h = x2 - x1, y2 - y1
    if h <= 1e-6:
        return GRIP_UNKNOWN
    aspect = w / h
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

# -- v2 pickup tunables -----------------------------------------------------
# After this many untouched frames while HELD, a fresh contact + outward motion
# is treated as a NEW pickup (re-grab), not continuity of the old hold.
# 26f0cc7c: ep5 (track7) pickup@263, silent until ~307, outward again 307-315.
REGRAB_SILENCE_FRAMES: int = 40
# Emergence-from-ROI pickup: depth rises from shelf (≤ EMERGE_FROM) to
# customer-side (≥ EMERGE_TO) during one contact burst.
EMERGE_FROM_DEPTH: float = 10.0
EMERGE_TO_DEPTH: float = 90.0
EMERGE_MIN_CONTACT: int = 3
EMERGE_MIN_RISE: float = 80.0  # bmax - bmin must cover a real lift
# Global gap between NEW pick confirmations (multi-hand HOI double-fires under
# looser final ROI) — real multi-item picks stay spaced ≥ ~30f on these cams.
MIN_PICKUP_EVENT_GAP: int = 25
# Team HAND_BUSY_GRACE is 90f — multi-item re-grabs on the same hand pay a
# permanent high bar. After this many frames without refreshing that carry
# stamp, treat the hand as free for the next pick (e7eea340 ~2nd pick).
HAND_BUSY_MULTIPICK_GRACE: int = 28
# Team CONTACT_K1=3; multipick lift-outs often only keep HF 2–4 frames once
# product clears the ROI mouth (e7ee ~201–205). Confirm with 2.
CONTACT_K1_V2: int = 2

# -- v2 putback tunables ----------------------------------------------------
PUTBACK_GRIP_CONFIRM: int = 3           # faster place signal (still needs streak)
PUTBACK_MIN_EVENT_GAP: int = 28         # successive putbacks (multi-item place cascade)
# Cross-hand debounce: HOI often splits one place across two hand tracks
# a few frames apart (7fdb ~255 + ~264 door). Per-hand gap alone cannot
# stop that; multi-item cascade still stays ≥ ~45–50f on 26f0/eeb.
PUTBACK_MIN_GLOBAL_EVENT_GAP: int = 28
PUTBACK_MIN_CARRY_FRAMES: int = 20      # after arm; first put can use pre-arm carry out
PUTBACK_MIN_CARRIED_DEPTH: float = 150.0
PUTBACK_MIN_FRAMES_OUT: int = 5
# Block place while still multi-item picking. e7eea340: false put@183 only 90f
# after pick@93 (re-entry mid-burst). 26f0 first real put is ~118f after last
# pick@392 — stay < that (use 110). Place stamp can be slightly earlier.
PUTBACK_MIN_AFTER_PICKUP: int = 110     # block place during multi-item pick burst
PUTBACK_MIN_PLACE_AFTER_PICKUP: int = 100  # place stamp itself must be after last pick
# Soft place with *no* product_contact_seen is disabled (df9c244: empty-hand
# re-pick was stamped as putback via SOFT_PLACE_MIN_AFTER_PICK alone). Occlusion
# still OK only through the remembered-product branch (product_contact_seen).
PUTBACK_SOFT_PLACE_MIN_AFTER_PICK: int = 160  # unused for stamp; kept for logs/compat
PUTBACK_CONTACT_OFF: int = 2            # frames without hf after shelf contact
PUTBACK_HAND_DEEP_FOR_CONTACT: float = 40.0  # slightly wider "at shelf"
PUTBACK_DEPTH_MEMORY: int = 90          # seed carry-out evidence before arm
PUTBACK_PLACE_MAX_AGE: int = 80         # allow full place-hold dwell before expire
PUTBACK_MIN_SHELF_DWELL: int = 5        # frames at shelf after place before retreat counts
PUTBACK_RETREAT_DEPTH: float = 100.0    # depth after leave counts as retreat
# After a putback, only briefly block new pickups (shelf noise during place).
# Permanent _put_phase blocked real re-picks on df9c244 (Issue C).
PUT_PHASE_BLOCK_FRAMES: int = 45
# Accept shelf contact if object is in_stable_roi OR simply in outer ROI OR
# hand is deep enough that the slot is occluded but contact is still real.

# -- machine / door FO reject (a77464a7 door touch + whole-machine HF) ------
# Absolute FO size vs 1080p frame (boxes like [540,799]–[1908,1073] ~18% area).
MACHINE_FO_MAX_AREA_FRAC: float = 0.10
MACHINE_FO_MAX_WIDTH_FRAC: float = 0.55
MACHINE_FO_MAX_ASPECT: float = 3.2  # door-strip ~5:1
# Hand-relative geometry (matches HOI-DETR relative checks intent)
REL_MAX_WIDTH_RATIO: float = 2.5
REL_MAX_HEIGHT_RATIO: float = 3.5
REL_MAX_SCALE_RATIO: float = 2.5  # linear; area uses scale^2
REL_MIN_VIOLATIONS_TO_REJECT: int = 2
# Door handle band: hand cy / bottom near image bottom (not product shelf)
# 0.78 ≈ bottom strip / machine FO y≈800 on 1080p (0.72 was too high and
# scrubbed legitimate deep shelf places as "door").
DOOR_HAND_CY_FRAC: float = 0.78
DOOR_HAND_Y2_FRAC: float = 0.92


def still_at_shelf_safe(depth_now: float) -> bool:
    return depth_now <= PUTBACK_HAND_DEEP_FOR_CONTACT + 30.0


# =============================================================================
# Pickup engine v2
# =============================================================================

class PickupEngineV2(PickupPutbackEngine):
    """Team pickup engine + latched re-grab after long silence (301-315 case)."""

    def __init__(self) -> None:
        super().__init__()
        self._last_touch_frame: Dict[int, int] = {}
        self._burst_min_depth: Dict[int, float] = {}
        self._burst_max_depth: Dict[int, float] = {}
        # latched until regrab confirms or ~45f after eligibility opens
        self._regrab_eligible: Dict[int, bool] = {}
        self._regrab_eligible_since: Dict[int, int] = {}
        self._last_pickup_frame: int = -10 ** 9
        # Briefly block new pickups after a putback (shelf noise during place
        # cascade). Timed — not permanent (df9c244 Issue C).
        self._put_phase: bool = False
        self._last_putback_frame: int = -10 ** 9

    def sync_putbacks(self, putbacks: List[dict]) -> None:
        """Feed putbacks so pickups are briefly deferred after a place."""
        for p in putbacks:
            f = int(p.get("frame", -1))
            if f < 0:
                continue
            self._put_phase = True
            self._last_putback_frame = max(self._last_putback_frame, f)

    def _should_block_new_pickup(self, frame_idx: int, kind: str = "normal") -> bool:
        # Put cascade debounce only for PUT_PHASE_BLOCK_FRAMES after last putback.
        if self._put_phase:
            if (frame_idx - self._last_putback_frame) > PUT_PHASE_BLOCK_FRAMES:
                self._put_phase = False
            else:
                logger.debug(
                    f"[PICKUP-v2] block {kind} frame={frame_idx} reason=put_phase "
                    f"last_pb={self._last_putback_frame}"
                )
                return True
        # Debounce multi-hand / double HOI confirms under wide ROI.
        if (frame_idx - self._last_pickup_frame) < MIN_PICKUP_EVENT_GAP:
            logger.debug(
                f"[PICKUP-v2] block {kind} frame={frame_idx} reason=min_gap "
                f"last_pu={self._last_pickup_frame}"
            )
            return True
        return False

    def _update_contact_signal(self, episode: Episode, link_conf: float, frame_idx: int) -> None:
        """Team logic with shorter CONTACT_K1 for multipick lift-outs."""
        if link_conf >= base.CONTACT_FLOOR:
            episode.contact_streak += 1
        else:
            episode.contact_streak = 0

        if episode.state == EP_RESTING and episode.contact_streak >= CONTACT_K1_V2:
            episode.state = EP_TRANSITIONING
            episode.contact_start_frame = frame_idx

    def _burst_emerged(self, episode_id: int, depth_now: float) -> bool:
        bmin = self._burst_min_depth.get(episode_id, 1e9)
        bmax = self._burst_max_depth.get(episode_id, -1e9)
        if bmax - bmin < EMERGE_MIN_RISE:
            return False
        if not (bmin <= EMERGE_FROM_DEPTH and bmax >= EMERGE_TO_DEPTH):
            return False
        # currently on the high side of the burst (lifting out), not deep-placing
        return depth_now >= (bmin + 0.55 * (bmax - bmin))

    def _confirm_pickup(self, episode: Episode, frame_idx: int) -> None:
        if self._should_block_new_pickup(frame_idx, kind="normal"):
            return
        super()._confirm_pickup(episode, frame_idx)
        # A confirmed pickup ends put-phase early (Issue C: allow later picks).
        self._put_phase = False
        eid = episode.episode_id
        self._last_touch_frame[eid] = frame_idx
        self._burst_min_depth[eid] = 1e9
        self._burst_max_depth[eid] = -1e9
        self._regrab_eligible[eid] = False
        self._regrab_eligible_since.pop(eid, None)
        self._last_pickup_frame = frame_idx

    def _maybe_regrab(self, episode: Episode, frame_idx: int, hand_obs: TrackObservation) -> bool:
        if episode.state != EP_HELD:
            return False
        if self._should_block_new_pickup(frame_idx, kind="regrab"):
            return False

        last = self._last_touch_frame.get(episode.episode_id, episode.last_transition_frame)
        # silence using last touch BEFORE this frame is recorded
        silence_before = frame_idx - last
        if silence_before >= REGRAB_SILENCE_FRAMES:
            self._regrab_eligible[episode.episode_id] = True
            self._regrab_eligible_since[episode.episode_id] = frame_idx

        if not self._regrab_eligible.get(episode.episode_id, False):
            return False

        since_elig = frame_idx - self._regrab_eligible_since.get(episode.episode_id, frame_idx)
        if since_elig > 45:
            self._regrab_eligible[episode.episode_id] = False
            return False

        depth_now = -hand_obs.safe_roi_distance
        emerged = self._burst_emerged(episode.episode_id, depth_now)
        enough_gap = (frame_idx - episode.last_transition_frame) >= MIN_TRANSITION_GAP
        if not (emerged and enough_gap and episode.contact_streak >= EMERGE_MIN_CONTACT):
            return False

        bmin = self._burst_min_depth.get(episode.episode_id, 0.0)
        bmax = self._burst_max_depth.get(episode.episode_id, 0.0)
        logger.info(
            f"[PICKUP-v2 regrab] episode={episode.episode_id} frame={frame_idx} "
            f"silence_before={silence_before}f track={episode.object_track_id} "
            f"depth_burst=[{bmin:.0f},{bmax:.0f}] outward={episode.outward_score:.1f}"
        )
        episode.last_transition_frame = frame_idx
        episode.outward_score = 0.0
        if episode.linked_hand_id is not None:
            self.hand_carrying[episode.linked_hand_id] = (episode.episode_id, frame_idx)
        self.pickups.append({
            "episode_id": episode.episode_id,
            "frame": frame_idx,
            "hand_track_id": episode.linked_hand_id,
            "object_track_id": episode.object_track_id,
            "kind": "regrab",
        })
        self.last_event_text = f"PICKUP CONFIRMED  episode={episode.episode_id} (regrab)"
        self.last_event_frame = frame_idx
        eid = episode.episode_id
        self._last_touch_frame[eid] = frame_idx
        self._burst_min_depth[eid] = 1e9
        self._burst_max_depth[eid] = -1e9
        self._regrab_eligible[eid] = False
        self._regrab_eligible_since.pop(eid, None)
        self._last_pickup_frame = frame_idx
        return True

    def _evaluate_transition(self, episode: Episode, frame_idx: int) -> None:
        if episode.state == EP_HELD:
            if episode.linked_hand_id is not None:
                self.hand_carrying[episode.linked_hand_id] = (episode.episode_id, frame_idx)
            return

        if episode.contact_start_frame is None:
            return
        since_contact = frame_idx - episode.contact_start_frame
        required_bar = graduated_bar(since_contact)
        enough_gap = (frame_idx - episode.last_transition_frame) >= MIN_TRANSITION_GAP

        if episode.state != EP_TRANSITIONING:
            return

        busy = self.hand_carrying.get(episode.linked_hand_id)
        if busy is not None and busy[0] != episode.episode_id:
            _, busy_since_frame = busy
            # Multipick: only apply busy bar if the prior hold was refreshed
            # recently. Stale HELD from an earlier bagged item must not block
            # the next product grab (e7ee ~f145 after pick@93).
            if (frame_idx - busy_since_frame) <= HAND_BUSY_MULTIPICK_GRACE:
                required_bar *= HAND_BUSY_BAR_MULTIPLIER
            elif (frame_idx - busy_since_frame) > HAND_BUSY_GRACE_FRAMES:
                # Truly stale — drop so later frames don't keep paying.
                self.hand_carrying.pop(episode.linked_hand_id, None)

        # No emerge-bypass for TRANSITIONING: place-direction bursts were
        # getting counted as pickups (512/565/713/772). Stick to outward_score.
        if episode.outward_score >= required_bar and enough_gap:
            self._confirm_pickup(episode, frame_idx)
        elif episode.contact_streak == 0 and episode.outward_score < NOISE_FLOOR:
            episode.state = EP_RESTING
            episode.outward_score = 0.0
            episode.contact_start_frame = None

    def process_frame(
        self,
        frame_data: Optional[dict],
        observations: List[TrackObservation],
        frame_idx: int,
    ) -> None:
        self._touched_episode_ids_this_frame = set()
        obs_by_track_id = {o.local_track_id: o for o in observations}
        live_fo_ids = {
            o.local_track_id for o in observations if o.class_name == "firstobject"
        }

        for hand_track_id, object_track_id, link_conf in resolve_hf_links_to_tracks(
            frame_data, observations
        ):
            if object_track_id is None:
                continue
            object_obs = obs_by_track_id.get(object_track_id)
            episode = self._find_or_create_episode(
                object_track_id, object_obs, frame_idx, live_fo_ids
            )
            episode.linked_hand_id = hand_track_id
            episode.frames_since_seen = 0
            self._touched_episode_ids_this_frame.add(episode.episode_id)

            hand_obs = obs_by_track_id.get(hand_track_id)
            if hand_obs is None:
                continue

            # Out-of-ROI new contact (lif t-out multipick only):
            # e7ee picks @148/@202 start with hand already customer-side.
            # Must stay out of put cascade (a774 shelf place was turning into
            # false PU@336/403 when exterior contact was always allowed).
            if not hand_obs.in_outer_roi and episode.state == EP_RESTING:
                depth_preview = -hand_obs.safe_roi_distance
                cold_start = (
                    episode.contact_streak <= 0 and episode.contact_start_frame is None
                )
                if cold_start:
                    in_multipick = (
                        not self._put_phase
                        and self._last_pickup_frame >= 0
                        and (frame_idx - self._last_pickup_frame) <= 120
                    )
                    if not in_multipick or depth_preview < EMERGE_TO_DEPTH:
                        continue

            depth_now = -hand_obs.safe_roi_distance
            prev_touch = self._last_touch_frame.get(episode.episode_id, -10 ** 9)
            if frame_idx - prev_touch > 5:
                self._burst_min_depth[episode.episode_id] = depth_now
                self._burst_max_depth[episode.episode_id] = depth_now
            else:
                self._burst_min_depth[episode.episode_id] = min(
                    self._burst_min_depth.get(episode.episode_id, depth_now), depth_now
                )
                self._burst_max_depth[episode.episode_id] = max(
                    self._burst_max_depth.get(episode.episode_id, depth_now), depth_now
                )

            self._update_contact_signal(episode, link_conf, frame_idx)
            self._update_motion_evidence(episode, hand_obs, object_obs, frame_idx)
            self._update_gesture_evidence(episode, hand_obs, frame_idx)

            if episode.state == EP_HELD:
                if not self._maybe_regrab(episode, frame_idx, hand_obs):
                    self.hand_carrying[hand_track_id] = (episode.episode_id, frame_idx)
            else:
                self._evaluate_transition(episode, frame_idx)

            self._last_touch_frame[episode.episode_id] = frame_idx

        # GC only — do NOT run team putback checks (owned by PutbackDetectorV2).
        self._handle_missing_evidence_pickup_only(frame_idx, obs_by_track_id)

    def _confirm_putback(self, episode: Episode, frame_idx: int) -> None:
        # Defensive: never let the inherited putback path arm a place event.
        return

    def _handle_missing_evidence_pickup_only(
        self,
        frame_idx: int,
        obs_by_track_id: Dict[int, TrackObservation],
    ) -> None:
        """Same housekeeping as team _handle_missing_evidence, minus putbacks."""
        for episode in list(self.episodes.values()):
            if episode.episode_id in self._touched_episode_ids_this_frame:
                continue
            episode.frames_since_seen += 1

            current_hand_id = None
            if episode.state == EP_HELD:
                current_hand_id = self._resolve_current_hand_track_id(
                    episode, obs_by_track_id, frame_idx
                )

            any_hand_visible = current_hand_id is not None or any(
                o.class_name == "hand" for o in obs_by_track_id.values()
            )
            if any_hand_visible:
                episode.frames_since_any_hand_seen = 0
            else:
                episode.frames_since_any_hand_seen += 1

            # pull optional SESSION_END constants if present
            end_frames = getattr(base, "SESSION_END_FRAMES", 300)
            rest_idle = getattr(base, "RESTING_IDLE_DROP_FRAMES", 120)
            trans_idle = getattr(base, "TRANSITIONING_IDLE_RESET_FRAMES", 40)

            if (
                episode.state == EP_HELD
                and episode.frames_since_seen > end_frames
                and episode.frames_since_any_hand_seen > end_frames
            ):
                self._resolve_as_kept(episode)
            elif (
                episode.state == EP_RESTING
                and episode.contact_streak == 0
                and episode.frames_since_seen > rest_idle
            ):
                self.episodes.pop(episode.object_track_id, None)
            elif (
                episode.state == EP_RESTING
                and episode.contact_streak > 0
                and episode.frames_since_seen > rest_idle
            ):
                episode.contact_streak = 0
            elif (
                episode.state == EP_TRANSITIONING
                and episode.frames_since_seen > trans_idle
            ):
                episode.state = EP_RESTING
                episode.contact_streak = 0
                episode.outward_score = 0.0
                episode.contact_start_frame = None


# =============================================================================
# Putback detector v2
# =============================================================================

@dataclass
class HandPutbackStateV2:
    hand_track_id: int
    grip_history: deque = field(default_factory=lambda: deque(maxlen=PUTBACK_GRIP_CONFIRM))
    # recent depths even while unarmed — first place must know we came from out
    depth_history: deque = field(default_factory=lambda: deque(maxlen=PUTBACK_DEPTH_MEMORY))
    carrying: bool = False
    carry_start_frame: Optional[int] = None
    max_depth_while_carrying: float = -1e9
    contacted_object_track: Optional[int] = None
    contact_frame: Optional[int] = None
    contact_depth: float = 0.0
    # first shelf return after carried-out (event time — not late confirm)
    place_frame: Optional[int] = None
    frames_since_contact_link: int = 0
    last_confirmed_frame: int = -PUTBACK_MIN_EVENT_GAP
    frames_carried_out: int = 0
    was_carried_out: bool = False
    release_armed: bool = False  # saw open/drop at shelf; waiting retreat to fire
    # True once a product-sized FO was contacted this put session (not machine FO)
    product_contact_seen: bool = False

    def seed_from_depth_memory(self) -> None:
        if not self.depth_history:
            return
        self.max_depth_while_carrying = max(self.max_depth_while_carrying, max(self.depth_history))
        out_frames = sum(1 for d in self.depth_history if d >= PUTBACK_MIN_CARRIED_DEPTH)
        if out_frames >= PUTBACK_MIN_FRAMES_OUT:
            self.was_carried_out = True
            self.frames_carried_out = max(self.frames_carried_out, out_frames)


class PutbackDetectorV2:
    """Putback-only detector:
      - arm carry only after pickup engine marks the hand busy (HELD)
      - shelf contact after a real carry-out (occlusion tolerant)
      - block place during multi-item pick phase (min frames after pickup)
    """

    def __init__(self) -> None:
        self.hand_states: Dict[int, HandPutbackStateV2] = {}
        self._hand_aspect_baseline: Dict[int, deque] = {}
        self.putbacks: List[dict] = []
        self.last_event_text: Optional[str] = None
        self.last_event_frame: int = -10 ** 9
        self._last_pickup_frame_by_hand: Dict[int, int] = {}
        self._last_pickup_frame_global: int = -10 ** 9
        # Last putback *event* frame across all hands (not per-track).
        self._last_putback_event_frame: int = -10 ** 9

    def sync_pickups(self, pickups: List[dict]) -> None:
        """Caller feeds confirmed pickups so putback waits until hold is real."""
        for p in pickups:
            f = int(p.get("frame", -1))
            hid = p.get("hand_track_id")
            if f < 0:
                continue
            self._last_pickup_frame_global = max(self._last_pickup_frame_global, f)
            if hid is not None:
                self._last_pickup_frame_by_hand[int(hid)] = max(
                    self._last_pickup_frame_by_hand.get(int(hid), -1), f
                )

    @staticmethod
    def _grip_confirmed(state: HandPutbackStateV2, reading: str) -> bool:
        recent = list(state.grip_history)
        return len(recent) >= PUTBACK_GRIP_CONFIRM and all(r == reading for r in recent)

    def _object_ok_for_contact(
        self,
        object_obs: Optional[TrackObservation],
        hand_depth: float,
        hand_obs: Optional[TrackObservation] = None,
    ) -> bool:
        if object_obs is None:
            # Occlusion: hand deep at shelf, object box gone — still allow
            # refreshing contact if we already have a contacted track id.
            return hand_depth <= PUTBACK_HAND_DEEP_FOR_CONTACT
        # Never treat whole-machine / door-strip FO as product contact
        if is_machine_scale_bbox(object_obs.bbox):
            return False
        if hand_obs is not None and not is_product_like_vs_hand(
            hand_obs.bbox, object_obs.bbox
        ):
            return False
        if object_obs.in_stable_roi or object_obs.in_outer_roi or object_obs.in_safe_roi:
            return True
        # object tracked outside polygon but hand is at the opening / shelf
        return hand_depth <= PUTBACK_HAND_DEEP_FOR_CONTACT + 40.0

    def _allow_place_stamp(
        self,
        hand_obs: TrackObservation,
        object_obs: Optional[TrackObservation],
        recalled_product: bool,
    ) -> bool:
        """Block door / out-of-ROI grips from stamping a putback place."""
        # Place = product into the machine slot → hand must be inside outer ROI.
        if not hand_obs.in_outer_roi:
            return False
        if object_obs is not None and is_product_like_observation(hand_obs, object_obs):
            return True
        if recalled_product:
            return not hand_in_door_band(hand_obs)
        return not hand_in_door_band(hand_obs)

    def process_frame(
        self,
        frame_data: Optional[dict],
        observations: List[TrackObservation],
        frame_idx: int,
        hand_carrying: Optional[Dict[int, Tuple[int, int]]] = None,
    ) -> None:
        obs_by_track_id = {o.local_track_id: o for o in observations}
        # best object per hand this frame (highest conf)
        touched: Dict[int, Tuple[int, float]] = {}
        for hand_id, object_id, conf in resolve_hf_links_to_tracks(frame_data, observations):
            if object_id is None:
                continue
            prev = touched.get(hand_id)
            if prev is None or conf > prev[1]:
                touched[hand_id] = (object_id, conf)

        active_hands: Set[int] = set()
        for hand_obs in observations:
            if hand_obs.class_name != "hand":
                continue
            hand_id = hand_obs.local_track_id
            active_hands.add(hand_id)
            state = self.hand_states.setdefault(hand_id, HandPutbackStateV2(hand_track_id=hand_id))
            baseline = self._hand_aspect_baseline.setdefault(
                hand_id, deque(maxlen=GESTURE_BASELINE_LEN)
            )
            state.grip_history.append(estimate_grip(hand_obs, baseline))
            depth_now = -hand_obs.safe_roi_distance
            state.depth_history.append(depth_now)

            # Prefer this hand's last pickup; fall back to any recent pickup
            # (hand track ids flip when cam loses the track mid-session).
            last_pu = self._last_pickup_frame_by_hand.get(
                hand_id, self._last_pickup_frame_global
            )
            # Also accept global last pickup so hand-rebinds still unlock puts.
            last_pu = max(last_pu, self._last_pickup_frame_global)
            hold_ready = (
                last_pu >= 0
                and (frame_idx - last_pu) >= PUTBACK_MIN_AFTER_PICKUP
            )

            if not state.carrying:
                # Arm when hold-ready and either closed grip (holding items)
                # OR hand is clearly back from outside approaching the shelf
                # (closed/unknown with seedable carry-out history).
                state.seed_from_depth_memory()
                # After track loss above ROI / off-cam, a new hand_id reappears
                # without depth memory — still treat as carried-out once past the
                # post-pick latency so first put (a774@336) is not lost.
                if (
                    hold_ready
                    and not state.was_carried_out
                    and last_pu >= 0
                    and (frame_idx - last_pu) >= PUTBACK_MIN_AFTER_PICKUP
                ):
                    state.was_carried_out = True
                    state.max_depth_while_carrying = max(
                        state.max_depth_while_carrying, PUTBACK_MIN_CARRIED_DEPTH
                    )
                    state.frames_carried_out = max(
                        state.frames_carried_out, PUTBACK_MIN_FRAMES_OUT
                    )
                approaching = (
                    state.was_carried_out
                    and depth_now <= PUTBACK_HAND_DEEP_FOR_CONTACT + 80.0
                )
                if hold_ready and (
                    self._grip_confirmed(state, GRIP_CLOSED) or approaching
                ):
                    state.carrying = True
                    state.carry_start_frame = frame_idx
                    state.seed_from_depth_memory()
                    state.max_depth_while_carrying = max(
                        state.max_depth_while_carrying, depth_now
                    )
                    logger.debug(
                        f"[TRACE-PUTBACK-v2] hand={hand_id} frame={frame_idx} now carrying "
                        f"max_depth={state.max_depth_while_carrying:.0f}"
                    )
                    # fall through same frame so place contact can stamp
                else:
                    continue

            # carrying
            state.max_depth_while_carrying = max(state.max_depth_while_carrying, depth_now)
            if depth_now >= PUTBACK_MIN_CARRIED_DEPTH:
                state.frames_carried_out += 1
                state.was_carried_out = True
            # Successive putbacks: short hops between shelf slots often never
            # reach MIN_CARRIED_DEPTH (150). After a confirmed put, re-arm with
            # a softer "left the place region" threshold so eeb@769 etc. stamp.
            elif (
                state.last_confirmed_frame >= 0
                and depth_now >= PUTBACK_RETREAT_DEPTH
            ):
                state.was_carried_out = True
                state.frames_carried_out = max(state.frames_carried_out, 1)
                state.max_depth_while_carrying = max(
                    state.max_depth_while_carrying, depth_now
                )

            pair = touched.get(hand_id)
            object_id = pair[0] if pair else None
            object_obs = obs_by_track_id.get(object_id) if object_id is not None else None

            at_shelf = depth_now <= PUTBACK_HAND_DEEP_FOR_CONTACT
            contact_ok = False
            allow_contact = (
                state.was_carried_out
                or state.max_depth_while_carrying >= PUTBACK_MIN_CARRIED_DEPTH
            )

            # Expire only truly stale place stamps. Never wipe place solely because
            # the hand is leaving — that is when place_retreat should fire
            # (eeb4886b puts ~712/769 were lost that way: leave cleared place
            # before grip-open could arm).
            if state.place_frame is not None:
                place_age = frame_idx - state.place_frame
                if place_age > PUTBACK_PLACE_MAX_AGE and not still_at_shelf_safe(
                    depth_now
                ):
                    # long gone from place region without a confirmed event
                    if not state.release_armed and place_age > PUTBACK_PLACE_MAX_AGE:
                        state.place_frame = None
                        state.contact_frame = None
                        state.contacted_object_track = None
                        state.release_armed = False
                        state.frames_since_contact_link = 0

            if allow_contact:
                live_ok = (
                    object_id is not None
                    and self._object_ok_for_contact(object_obs, depth_now, hand_obs)
                )
                if live_ok:
                    contact_ok = True
                    state.contacted_object_track = object_id
                    if is_product_like_observation(hand_obs, object_obs):
                        state.product_contact_seen = True
                    # Stamp place once hand returns to shelf after carry-out,
                    # and only once far enough past the last pickup (not mid-pick).
                    if (
                        at_shelf
                        and state.was_carried_out
                        and (frame_idx - last_pu) >= PUTBACK_MIN_PLACE_AFTER_PICKUP
                        and self._allow_place_stamp(
                            hand_obs, object_obs, state.product_contact_seen
                        )
                    ):
                        if state.place_frame is None:
                            state.place_frame = frame_idx
                            state.contact_frame = frame_idx
                    elif state.contact_frame is None and at_shelf:
                        state.contact_frame = frame_idx
                    state.contact_depth = depth_now
                    state.frames_since_contact_link = 0
                elif (
                    state.contacted_object_track is not None
                    and state.product_contact_seen
                    and at_shelf
                    and object_id is None
                    and self._allow_place_stamp(hand_obs, None, True)
                ):
                    # Remembered *product* contact + hand deep: common put path
                    # (object box drops at place). Never recall machine FO.
                    state.frames_since_contact_link = 0
                    contact_ok = True
                    if (
                        state.was_carried_out
                        and (frame_idx - last_pu) >= PUTBACK_MIN_PLACE_AFTER_PICKUP
                        and state.place_frame is None
                    ):
                        state.place_frame = frame_idx
                        if state.contact_frame is None:
                            state.contact_frame = frame_idx
                    state.contact_depth = depth_now
                elif (
                    at_shelf
                    and state.was_carried_out
                    and depth_now <= PUTBACK_HAND_DEEP_FOR_CONTACT
                    and (frame_idx - last_pu) >= PUTBACK_MIN_PLACE_AFTER_PICKUP
                    # Issue B: never soft-place on empty hands. Require a product
                    # seen this put session (live FO path / remembered-product
                    # branch above). Time-since-pickup alone caused false puts.
                    and state.product_contact_seen
                    and self._allow_place_stamp(
                        hand_obs, object_obs, state.product_contact_seen
                    )
                ):
                    # Soft place only after product was seen this put session
                    # (object box may have dropped; oid may be gone).
                    contact_ok = True
                    if state.contacted_object_track is None:
                        state.contacted_object_track = -1
                    if state.place_frame is None:
                        state.place_frame = frame_idx
                        state.contact_frame = frame_idx
                    state.contact_depth = depth_now
                    state.frames_since_contact_link = 0
                else:
                    state.frames_since_contact_link += 1
            else:
                state.frames_since_contact_link += 1

            # Door-band grip without product: kill false place (a774 door).
            # If product_contact_seen, keep the place while the hand is still
            # deep in the slot (shelf bottom ↔ door band overlap on cam1 ROI).
            if (
                state.place_frame is not None
                and hand_in_door_band(hand_obs)
                and not state.product_contact_seen
            ):
                state.place_frame = None
                state.contact_frame = None
                state.contacted_object_track = None
                state.release_armed = False
                state.frames_since_contact_link = 0
                continue

            # Drop place stamps that began too soon after a pickup (pick re-entry)
            if (
                state.place_frame is not None
                and last_pu >= 0
                and (state.place_frame - last_pu) < PUTBACK_MIN_PLACE_AFTER_PICKUP
            ):
                state.place_frame = None
                state.contact_frame = None
                state.contacted_object_track = None
                state.release_armed = False
                state.frames_since_contact_link = 0
                continue

            if state.contacted_object_track is None and state.place_frame is None:
                continue

            # "carried enough" uses lifetime of this putback-session (carry_start),
            # not inter-putback re-arm delay — putbacks in sequence stay armed.
            carry_frames = (
                frame_idx - state.carry_start_frame
                if state.carry_start_frame is not None else 0
            )
            frames_since_last_pb = frame_idx - state.last_confirmed_frame
            place_age = (
                frame_idx - state.place_frame
                if state.place_frame is not None else -1
            )
            # Allow a new place stamp even slightly before MIN_EVENT_GAP elapses
            # once a fresh shelf entry began after the previous putback event.
            fresh_place = (
                state.place_frame is not None
                and state.place_frame > state.last_confirmed_frame
                and place_age >= PUTBACK_MIN_SHELF_DWELL
            )
            carried_enough = (
                hold_ready
                and (
                    state.max_depth_while_carrying >= PUTBACK_MIN_CARRIED_DEPTH
                    or (
                        state.last_confirmed_frame >= 0
                        and state.was_carried_out
                        and state.max_depth_while_carrying >= PUTBACK_RETREAT_DEPTH
                    )
                )
                and (
                    state.frames_carried_out >= PUTBACK_MIN_FRAMES_OUT
                    or state.was_carried_out
                )
                and (
                    (state.last_confirmed_frame < 0 and carry_frames >= PUTBACK_MIN_CARRY_FRAMES)
                    or (state.last_confirmed_frame >= 0 and frames_since_last_pb >= PUTBACK_MIN_EVENT_GAP)
                    or fresh_place
                    or (
                        state.last_confirmed_frame < 0
                        and state.was_carried_out
                        and state.place_frame is not None
                    )
                )
            )

            grip_open = self._grip_confirmed(state, GRIP_OPEN)
            link_dropped = state.frames_since_contact_link >= PUTBACK_CONTACT_OFF
            still_at_place = depth_now <= PUTBACK_HAND_DEEP_FOR_CONTACT + 30.0
            retreated_out = (
                depth_now > PUTBACK_RETREAT_DEPTH
                and state.place_frame is not None
                and place_age >= PUTBACK_MIN_SHELF_DWELL
            )

            # Arm release once we see place + open/drop while still at shelf.
            if (
                state.place_frame is not None
                and carried_enough
                and (grip_open or link_dropped)
                and still_at_place
            ):
                state.release_armed = True

            # Also arm after a dwell on shelf even without clean grip (eeb places
            # often show CLOSED→UNKNOWN while deep in the machine).
            if (
                state.place_frame is not None
                and carried_enough
                and still_at_place
                and place_age >= PUTBACK_MIN_SHELF_DWELL
            ):
                state.release_armed = True

            released = False
            reason = ""
            # Pure door-handle pose: allow release only if we saw a real product
            # on this put session (then the hand often dips lower after place).
            door_now = hand_in_door_band(hand_obs) and not state.product_contact_seen
            # Issue B: every confirm path needs product_contact_seen this put
            # (blocks empty-hand re-pick → false PUTBACK on df9c244).
            if (
                door_now
                or not carried_enough
                or state.place_frame is None
                or not state.product_contact_seen
            ):
                released = False
            elif retreated_out and (
                state.release_armed or place_age >= PUTBACK_MIN_SHELF_DWELL
            ):
                # core put path: visited shelf then left (doesn't require open grip)
                released = True
                reason = "place_retreat"
            elif (
                state.release_armed
                and still_at_place
                and place_age >= 18
            ):
                released = True
                reason = "place_hold"
            elif (
                link_dropped
                and still_at_place
                and not contact_ok
                and place_age >= 10
            ):
                released = True
                reason = "link_drop_at_shelf"

            gap_ok = (
                (frame_idx - state.last_confirmed_frame) >= PUTBACK_MIN_EVENT_GAP
                or (
                    state.place_frame is not None
                    and state.place_frame > state.last_confirmed_frame
                    and place_age >= PUTBACK_MIN_SHELF_DWELL
                )
            )
            if released and gap_ok:
                place_age_evt = (
                    frame_idx - state.place_frame
                    if state.place_frame is not None else 10 ** 9
                )
                # Prefer retreat time, else recent place, else confirm frame.
                # Never keep a multi-second-old place stamp as the event time.
                if reason == "place_retreat":
                    event_frame = frame_idx
                elif state.place_frame is not None and place_age_evt <= PUTBACK_PLACE_MAX_AGE:
                    event_frame = state.place_frame
                else:
                    event_frame = frame_idx

                # Cross-hand debounce (two hand tracks, one physical place).
                g_last = self._last_putback_event_frame
                global_gap_ok = (
                    (event_frame - g_last) >= PUTBACK_MIN_GLOBAL_EVENT_GAP
                    and (frame_idx - g_last) >= PUTBACK_MIN_GLOBAL_EVENT_GAP
                )
                if not global_gap_ok:
                    logger.debug(
                        f"[PUTBACK-v2] suppress hand={hand_id} event={event_frame} "
                        f"confirm={frame_idx} last_global={g_last} reason=global_gap"
                    )
                    state.place_frame = None
                    state.contact_frame = None
                    state.contacted_object_track = None
                    state.release_armed = False
                    state.frames_since_contact_link = 0
                    state.product_contact_seen = False
                    continue

                oid = state.contacted_object_track
                if oid is not None and oid < 0:
                    oid = None
                self.putbacks.append({
                    "hand_track_id": hand_id,
                    "object_track_id": oid,
                    "contact_frame": state.contact_frame,
                    "place_frame": state.place_frame,
                    "frame": event_frame,
                    "confirm_frame": frame_idx,
                    "reason": reason,
                    "max_carried_depth": round(state.max_depth_while_carrying, 1),
                })
                logger.info(
                    f"[PUTBACK-v2] hand={hand_id} object={oid} "
                    f"place_frame={state.place_frame} confirm_frame={frame_idx} "
                    f"event_frame={event_frame} reason={reason}"
                )
                self.last_event_text = f"PUTBACK CONFIRMED  hand={hand_id}"
                self.last_event_frame = event_frame
                self._last_putback_event_frame = max(event_frame, frame_idx)
                state.last_confirmed_frame = frame_idx
                state.contacted_object_track = None
                state.contact_frame = None
                state.place_frame = None
                state.release_armed = False
                state.frames_since_contact_link = 0
                state.product_contact_seen = False
                state.grip_history.clear()
                # Sibling hand tracks often mirror the same place → wipe their
                # pending place so a second put cannot fire ~10f later (7fdb).
                for other_id, other in self.hand_states.items():
                    if other_id == hand_id:
                        continue
                    other.place_frame = None
                    other.contact_frame = None
                    other.contacted_object_track = None
                    other.release_armed = False
                    other.product_contact_seen = False
                    other.frames_since_contact_link = 0
                    other.last_confirmed_frame = max(
                        other.last_confirmed_frame, frame_idx
                    )
                # Always require leave + re-enter after a confirm. Camping on the
                # shelf (place_hold) must not keep was_carried_out true, or the
                # next item never gets a fresh place stamp (miss eeb@769).
                state.was_carried_out = depth_now >= PUTBACK_MIN_CARRIED_DEPTH
                if still_at_place and not state.was_carried_out:
                    state.was_carried_out = False
                state.frames_carried_out = 1 if state.was_carried_out else 0
                state.max_depth_while_carrying = depth_now
                state.depth_history.clear()
                if state.was_carried_out:
                    state.depth_history.append(depth_now)

        # GC idle hands
        for hid in list(self.hand_states.keys()):
            if hid not in active_hands:
                # keep state a bit? drop after long absence
                pass


# Backward-compatible alias used by older imports
PickupPutbackEngineV2 = PickupEngineV2

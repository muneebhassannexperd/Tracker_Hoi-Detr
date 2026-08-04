"""
Hybrid-SORT / Hybrid-SORT-ReID adapter for HOI-DETR.

Wraps ymzis69/HybridSORT (`trackers/hybrid_sort_tracker`).

Deep Hybrid SORT (= Hybrid-SORT-ReID) uses HOI-DETR decoder embeddings as
appearance features (no separate FastReID checkpoint required). Plain
Hybrid-SORT (TCM + weak cues, no ReID) is also available.

Tracks each HOI class separately (hand / firstobject / secondobject)
so IDs never cross class boundaries.

ID stability:
  - Stable IDs are minted independently of Hybrid-SORT's internal counter
    (so Hybrid recycling Kalman id=9 never forces label #9 onto a new object)
  - Last-frame continuity keeps an ID through brief detection dropouts
  - Bank reclaim of passed IDs is OFF by default (avoids #9 recycle on pickups)
  - Size/appearance jumps on a continuing Hybrid tracklet mint a new ID
    (machine shelf contact → picked product)
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import BaseTracker

_REPO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "external", "HybridSORT")
)
if _REPO in sys.path:
    sys.path.remove(_REPO)
sys.path.insert(0, _REPO)

from trackers.hybrid_sort_tracker.association import iou_batch  # noqa: E402
from trackers.hybrid_sort_tracker.hybrid_sort import Hybrid_Sort  # noqa: E402
from trackers.hybrid_sort_tracker.hybrid_sort_reid import (  # noqa: E402
    Hybrid_Sort_ReID,
)


def _match_tracks_to_dets(dets, tracks, iou_thr=0.3, soft_iou_thr=0.01):
    """
    Greedy IoU match of tracker outputs (Nx5: xyxy + id) back onto dets.

    Two-pass: strict threshold first, then a soft pass so detections that
    Hybrid-SORT did associate (but with a slightly drifted Kalman box) still
    receive an ID instead of falling back to track_id=None.
    """
    ids = [None] * len(dets)
    if len(dets) == 0 or tracks is None or len(tracks) == 0:
        return ids

    det_boxes = np.asarray([d["box"] for d in dets], dtype=np.float32)
    trk_boxes = tracks[:, :4].astype(np.float32)
    trk_ids = tracks[:, 4].astype(np.int32)

    ious = iou_batch(det_boxes, trk_boxes)  # [N_det, N_trk]
    used_trks = set()

    def _assign(thr):
        pairs = [
            (float(ious[i, j]), i, j)
            for i in range(ious.shape[0])
            for j in range(ious.shape[1])
            if ious[i, j] >= thr
        ]
        pairs.sort(reverse=True)
        for _, di, tj in pairs:
            if ids[di] is not None or tj in used_trks:
                continue
            ids[di] = int(trk_ids[tj])
            used_trks.add(tj)

    _assign(iou_thr)
    if soft_iou_thr < iou_thr:
        _assign(soft_iou_thr)
    return ids


def _l2_normalize(feat: np.ndarray) -> np.ndarray:
    feat = np.asarray(feat, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(feat))
    if n < 1e-12:
        return feat
    return feat / n


def _box_area(box) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def _area_ratio_ok(box_a, box_b, lo: float = 0.45, hi: float = 2.25) -> bool:
    """True if box areas are similar enough to be the same instance."""
    a = _box_area(box_a)
    b = _box_area(box_b)
    if a < 1.0 or b < 1.0:
        return False
    r = a / b
    return lo <= r <= hi


def _box_center(box) -> Tuple[float, float]:
    return (0.5 * (float(box[0]) + float(box[2])),
            0.5 * (float(box[1]) + float(box[3])))


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


class InactiveTrackBank:
    """
    Hold recently-lost track identities so a reappearing *same* instance can
    reclaim its ID. Reclaim requires a real spatial+appearance match — we do
    NOT auto-assign a lone lost ID to whichever new detection appears next
    (that caused new hands/products to steal old IDs).
    """

    def __init__(
        self,
        hold_frames: int = 120,
        sim_thresh: float = 0.55,
        iou_thresh: float = 0.12,
        max_center_dist: float = 150.0,
        # Stricter continuity gate for same-frame dropout fill (prev_stable).
        prev_iou_thresh: float = 0.08,
        prev_max_center_dist: float = 100.0,
    ):
        self.hold_frames = hold_frames
        self.sim_thresh = sim_thresh
        self.iou_thresh = iou_thresh
        self.max_center_dist = max_center_dist
        self.prev_iou_thresh = prev_iou_thresh
        self.prev_max_center_dist = prev_max_center_dist
        self._items: List[dict] = []

    def clear(self) -> None:
        self._items = []

    def prune(self, frame_idx: int) -> None:
        self._items = [
            it for it in self._items
            if frame_idx - it["lost_frame"] <= self.hold_frames
        ]

    def items_for_class(self, class_id, used_ids, frame_idx) -> List[dict]:
        self.prune(frame_idx)
        out = []
        for it in self._items:
            if it["class_id"] != int(class_id):
                continue
            if it["track_id"] in used_ids:
                continue
            out.append(it)
        return out

    def push(self, track_id, class_id, feat, box, frame_idx) -> None:
        # Replace any older entry for the same ID (keep freshest appearance).
        self._items = [it for it in self._items if it["track_id"] != track_id]
        self._items.append({
            "track_id": int(track_id),
            "class_id": int(class_id),
            "feat": _l2_normalize(feat) if feat is not None else None,
            "box": np.asarray(box, dtype=np.float32)[:4].copy(),
            "lost_frame": int(frame_idx),
        })

    def pop(self, track_id) -> None:
        self._items = [it for it in self._items if it["track_id"] != track_id]

    def _geometry(self, box, it):
        cx, cy = _box_center(box)
        ix, iy = _box_center(it["box"])
        iou = _box_iou(box, it["box"])
        dist = ((cx - ix) ** 2 + (cy - iy) ** 2) ** 0.5
        return iou, dist

    def _similarity(self, feat, it) -> float:
        if feat is None or it.get("feat") is None:
            return 0.0
        return float(np.dot(_l2_normalize(feat), it["feat"]))

    def accepts(self, feat, box, it, mode: str = "bank") -> bool:
        """
        mode='prev': near-continuous dropout fill (last frame only).
        mode='bank': rebirth after a gap — spatial + appearance + similar size.
        """
        iou, dist = self._geometry(box, it)
        sim = self._similarity(feat, it)
        area_ok = _area_ratio_ok(box, it["box"])

        if mode == "prev":
            # Same instance almost certainly still in place.
            # Also reject huge size jumps (machine shelf ↔ small product).
            if not area_ok:
                return False
            return iou >= self.prev_iou_thresh or dist <= self.prev_max_center_dist

        # Bank rebirth: reject weak / cross-instance matches.
        if not area_ok:
            return False
        spatial_ok = iou >= self.iou_thresh or dist <= self.max_center_dist
        if feat is None or it.get("feat") is None:
            return iou >= 0.25 or dist <= 60.0
        return spatial_ok and sim >= self.sim_thresh

    def score_match(self, feat, box, it, frame_idx) -> float:
        """Higher is better among candidates that already passed accepts()."""
        iou, dist = self._geometry(box, it)
        sim = self._similarity(feat, it)
        spat = max(iou, 1.0 - min(dist / max(self.max_center_dist, 1.0), 1.0))
        age_pen = 0.001 * max(0, frame_idx - it["lost_frame"])
        return 0.55 * sim + 0.45 * spat - age_pen

    def query(self, class_id, feat, box, frame_idx, used_ids) -> Optional[int]:
        cands = self.items_for_class(class_id, used_ids, frame_idx)
        if not cands:
            return None

        best_id = None
        best_score = -1e9
        for it in cands:
            if not self.accepts(feat, box, it, mode="bank"):
                continue
            score = self.score_match(feat, box, it, frame_idx)
            if score > best_score:
                best_score = score
                best_id = it["track_id"]

        if best_id is not None:
            self.pop(best_id)
        return best_id


def _candidate_pool(prev_stable, bank, class_id, used_ids, frame_idx, sources):
    """
    Build candidate ID list.
    sources: iterable containing 'prev' and/or 'bank'.
    """
    pool = []
    if "prev" in sources:
        for sid, info in prev_stable.items():
            if sid in used_ids:
                continue
            if int(info["class_id"]) != int(class_id):
                continue
            pool.append({
                "track_id": int(sid),
                "class_id": int(class_id),
                "feat": info.get("feat"),
                "box": info["box"],
                "lost_frame": frame_idx,
                "source": "prev",
            })
    if "bank" in sources:
        for it in bank.items_for_class(class_id, used_ids, frame_idx):
            pool.append({**it, "source": "bank"})
    return pool


def _pick_from_pool(pool, feat, box, frame_idx, bank) -> Optional[int]:
    """Pick best accepted candidate; no singleton auto-assign."""
    if not pool:
        return None

    best_id = None
    best_score = -1e9
    for it in pool:
        mode = "prev" if it.get("source") == "prev" else "bank"
        if not bank.accepts(feat, box, it, mode=mode):
            continue
        score = bank.score_match(feat, box, it, frame_idx)
        # Prefer prev-frame continuity slightly over bank rebirth.
        if mode == "prev":
            score += 0.05
        if score > best_score:
            best_score = score
            best_id = it["track_id"]
    return best_id



def _default_args(
    det_thresh=0.3,
    use_byte=True,
    tcm_first_step=True,
    tcm_byte_step=True,
    tcm_first_step_weight=1.0,
    tcm_byte_step_weight=1.0,
    eg_weight_high_score=1.3,
    eg_weight_low_score=1.2,
    low_thresh=0.1,
    high_score_matching_thresh=0.8,
    low_score_matching_thresh=0.5,
    alpha=0.8,
    with_longterm_reid=True,
    with_longterm_reid_correction=True,
    longterm_reid_weight=0.2,
    longterm_reid_weight_low=0.15,
    longterm_reid_correction_thresh=0.8,
    longterm_reid_correction_thresh_low=0.85,
    longterm_bank_length=60,
    adapfs=False,
    ecc=False,
):
    """Args namespace tuned for HOI ID stability (long memory + long-term ReID)."""
    return SimpleNamespace(
        use_byte=use_byte,
        dataset="hoi",
        track_thresh=det_thresh,
        kalman_GPR=False,
        TCM_first_step=tcm_first_step,
        TCM_byte_step=tcm_byte_step,
        TCM_first_step_weight=tcm_first_step_weight,
        TCM_byte_step_weight=tcm_byte_step_weight,
        EG_weight_high_score=eg_weight_high_score,
        EG_weight_low_score=eg_weight_low_score,
        low_thresh=low_thresh,
        high_score_matching_thresh=high_score_matching_thresh,
        low_score_matching_thresh=low_score_matching_thresh,
        alpha=alpha,
        with_longterm_reid=with_longterm_reid,
        with_longterm_reid_correction=with_longterm_reid_correction,
        longterm_reid_weight=longterm_reid_weight,
        longterm_reid_weight_low=longterm_reid_weight_low,
        longterm_reid_correction_thresh=longterm_reid_correction_thresh,
        longterm_reid_correction_thresh_low=longterm_reid_correction_thresh_low,
        longterm_bank_length=longterm_bank_length,
        adapfs=adapfs,
        ECC=ecc,
    )


class HybridSortTracker(BaseTracker):
    """
    Per-class Hybrid-SORT tracker with inactive-ID recovery.

    with_reid=True  -> Hybrid-SORT-ReID (Deep Hybrid SORT), using HOI embeds
    with_reid=False -> Hybrid-SORT (TCM / weak cues only)
    """

    def __init__(
        self,
        det_thresh: float = 0.3,
        max_age: int = 90,
        min_hits: int = 1,
        iou_threshold: float = 0.15,
        match_iou: float = 0.1,
        delta_t: int = 3,
        inertia: float = 0.05,
        asso_func: str = "Height_Modulated_IoU",
        with_reid: bool = True,
        use_byte: bool = True,
        tcm_first_step: bool = True,
        tcm_byte_step: bool = True,
        tcm_first_step_weight: float = 1.0,
        tcm_byte_step_weight: float = 1.0,
        eg_weight_high_score: float = 1.5,
        eg_weight_low_score: float = 1.3,
        with_longterm_reid: bool = True,
        with_longterm_reid_correction: bool = True,
        longterm_reid_weight: float = 0.2,
        longterm_reid_weight_low: float = 0.15,
        longterm_bank_length: int = 60,
        recover_hold_frames: int = 60,
        recover_sim_thresh: float = 0.65,
        recover_max_center_dist: float = 120.0,
        # If False (default), passed-object IDs are never reused from the bank.
        # Only last-frame continuity can keep an ID. Prevents #9 recycling onto
        # a new pickup after an earlier product already used #9.
        allow_bank_reclaim: bool = False,
        # Split a continuing Hybrid track when box size / appearance jumps
        # (e.g. large machine shelf box → small product box).
        split_on_instance_change: bool = True,
        area_ratio_lo: float = 0.45,
        area_ratio_hi: float = 2.25,
        split_sim_thresh: float = 0.35,
        class_ids=(0, 1, 2),
        feat_dim: Optional[int] = None,
    ):
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.match_iou = match_iou
        self.delta_t = delta_t
        self.inertia = inertia
        self.asso_func = asso_func
        self.with_reid = with_reid
        self.class_ids = tuple(class_ids)
        self.feat_dim = feat_dim  # filled on first ReID update if None
        self.allow_bank_reclaim = allow_bank_reclaim
        self.split_on_instance_change = split_on_instance_change
        self.area_ratio_lo = area_ratio_lo
        self.area_ratio_hi = area_ratio_hi
        self.split_sim_thresh = split_sim_thresh
        self.args = _default_args(
            det_thresh=det_thresh,
            use_byte=use_byte,
            tcm_first_step=tcm_first_step,
            tcm_byte_step=tcm_byte_step,
            tcm_first_step_weight=tcm_first_step_weight,
            tcm_byte_step_weight=tcm_byte_step_weight,
            eg_weight_high_score=eg_weight_high_score if with_reid else 0.0,
            eg_weight_low_score=eg_weight_low_score if with_reid else 0.0,
            with_longterm_reid=with_longterm_reid and with_reid,
            with_longterm_reid_correction=with_longterm_reid_correction and with_reid,
            longterm_reid_weight=longterm_reid_weight if with_reid else 0.0,
            longterm_reid_weight_low=longterm_reid_weight_low if with_reid else 0.0,
            longterm_bank_length=longterm_bank_length,
        )
        self.name = "hybrid_sort_reid" if with_reid else "hybrid_sort"
        self._bank = InactiveTrackBank(
            hold_frames=recover_hold_frames,
            sim_thresh=recover_sim_thresh,
            max_center_dist=recover_max_center_dist,
        )
        self._trackers: Dict[int, object] = {}
        self._hybrid_to_stable: Dict[Tuple[int, int], int] = {}
        self._prev_stable: Dict[int, dict] = {}
        self._next_stable_id = 1
        self._frame_idx = 0
        self.reset()

    def _make_tracker(self):
        kwargs = dict(
            args=self.args,
            det_thresh=self.det_thresh,
            max_age=self.max_age,
            min_hits=self.min_hits,
            iou_threshold=self.iou_threshold,
            delta_t=self.delta_t,
            asso_func=self.asso_func,
            inertia=self.inertia,
        )
        if self.with_reid:
            return Hybrid_Sort_ReID(**kwargs)
        return Hybrid_Sort(use_byte=self.args.use_byte, **kwargs)

    def reset(self) -> None:
        self._trackers = {cid: self._make_tracker() for cid in self.class_ids}
        self._hybrid_to_stable = {}
        self._prev_stable = {}
        self._next_stable_id = 1
        self._frame_idx = 0
        self._bank.clear()

    def _alloc_id(self, used_stable) -> int:
        """Mint a fresh stable ID never tied to Hybrid-SORT's internal counter."""
        while (
            self._next_stable_id in used_stable
            or self._next_stable_id in self._prev_stable
            or any(it["track_id"] == self._next_stable_id for it in self._bank._items)
        ):
            self._next_stable_id += 1
        sid = self._next_stable_id
        self._next_stable_id += 1
        return sid

    def _features_for(self, cls_dets):
        """Build (N, D) appearance matrix from HOI embeddings on dets."""
        if not cls_dets:
            dim = self.feat_dim or 256
            return np.zeros((0, dim), dtype=np.float32)

        feats = []
        for d in cls_dets:
            emb = d.get("embedding", None)
            if emb is None:
                raise ValueError(
                    "Hybrid-SORT-ReID requires detection['embedding']. "
                    "Attach HOI decoder embeds before tracker.update()."
                )
            feats.append(_l2_normalize(emb))
        arr = np.stack(feats, axis=0)
        self.feat_dim = arr.shape[1]
        return arr

    def _is_instance_change(self, prev_info, feat, box) -> bool:
        """
        Detect machine-shelf → product (or any clear instance swap) on a
        continuing Hybrid track.
        """
        if prev_info is None:
            return False
        prev_box = prev_info.get("box")
        if prev_box is None:
            return False
        if not _area_ratio_ok(
            box, prev_box, lo=self.area_ratio_lo, hi=self.area_ratio_hi
        ):
            return True
        prev_feat = prev_info.get("feat")
        if feat is not None and prev_feat is not None:
            sim = float(np.dot(_l2_normalize(feat), _l2_normalize(prev_feat)))
            iou = _box_iou(box, prev_box)
            # Appearance collapsed while only weakly overlapping → new instance.
            if sim < self.split_sim_thresh and iou < 0.20:
                return True
        return False

    def _best_recover_id(self, class_id, feat, box, used_ids, pool, sources=("prev",)):
        """
        Reclaim only via last-frame continuity by default.
        Bank reclaim is opt-in and still requires strong same-instance evidence.
        """
        cands = _candidate_pool(
            pool, self._bank, class_id, used_ids, self._frame_idx, sources
        )
        sid = _pick_from_pool(cands, feat, box, self._frame_idx, self._bank)
        if sid is not None:
            self._bank.pop(sid)
        return sid

    def _stabilize_ids(self, detections: List[dict]) -> None:
        """
        Map Hybrid-SORT internal IDs onto stable IDs.

        Critical: stable IDs are NOT copied from Hybrid's track counter.
        Hybrid reusing internal id=9 after an old object passed must not
        force stable id=9 onto a new product.
        """
        used_stable = set()
        curr_stable: Dict[int, dict] = {}
        bank_sources = ("prev", "bank") if self.allow_bank_reclaim else ("prev",)

        # Pass 1: detections that Hybrid-SORT matched.
        for d in detections:
            hid = d.get("track_id")
            if hid is None:
                continue
            cid = int(d["class_id"])
            key = (cid, int(hid))
            feat = d.get("embedding", None)
            box = d["box"]

            sid = None
            if key in self._hybrid_to_stable:
                candidate = self._hybrid_to_stable[key]
                if candidate not in used_stable:
                    prev = self._prev_stable.get(candidate)
                    if (
                        self.split_on_instance_change
                        and self._is_instance_change(prev, feat, box)
                    ):
                        # Same Hybrid tracklet, different physical instance.
                        # Block the abandoned ID from being reused this frame.
                        used_stable.add(candidate)
                        sid = None
                    else:
                        sid = candidate

            if sid is None:
                recovered = self._best_recover_id(
                    class_id=cid,
                    feat=feat,
                    box=box,
                    used_ids=used_stable,
                    pool=self._prev_stable,
                    sources=bank_sources,
                )
                # Reject recovered ID if size/appearance says new instance.
                if (
                    recovered is not None
                    and self.split_on_instance_change
                ):
                    prev = self._prev_stable.get(recovered)
                    if prev is None:
                        for it in self._bank._items:
                            if it["track_id"] == recovered:
                                prev = it
                                break
                    if self._is_instance_change(prev, feat, box):
                        recovered = None

                if recovered is not None:
                    sid = recovered
                else:
                    sid = self._alloc_id(used_stable)

            self._hybrid_to_stable[key] = sid
            d["track_id"] = sid
            used_stable.add(sid)
            curr_stable[sid] = {
                "class_id": cid,
                "feat": feat,
                "box": box,
            }

        # Pass 2: dropout frames — keep previous ID only if still nearby
        # and size-consistent (no machine→product fill).
        for d in detections:
            if d.get("track_id") is not None:
                continue
            cid = int(d["class_id"])
            feat = d.get("embedding", None)
            box = d["box"]
            sid = self._best_recover_id(
                class_id=cid,
                feat=feat,
                box=box,
                used_ids=used_stable,
                pool=self._prev_stable,
                sources=("prev",),
            )
            if sid is None:
                continue
            prev = self._prev_stable.get(sid)
            if self.split_on_instance_change and self._is_instance_change(
                prev, feat, box
            ):
                continue
            d["track_id"] = sid
            used_stable.add(sid)
            curr_stable[sid] = {
                "class_id": cid,
                "feat": feat,
                "box": box,
            }

        # Tracks active last frame but missing now → park in inactive bank
        # (used only if allow_bank_reclaim=True).
        for sid, info in self._prev_stable.items():
            if sid not in curr_stable:
                self._bank.push(
                    track_id=sid,
                    class_id=info["class_id"],
                    feat=info.get("feat"),
                    box=info["box"],
                    frame_idx=self._frame_idx,
                )

        active_sids = set(curr_stable.keys())
        self._hybrid_to_stable = {
            k: v for k, v in self._hybrid_to_stable.items() if v in active_sids
        }

        self._prev_stable = curr_stable
        self._frame_idx += 1

    def update(self, detections: List[dict], frame) -> List[dict]:
        if frame is None:
            for d in detections:
                d["track_id"] = None
            return detections

        h, w = frame.shape[:2]
        # Boxes are already in original-frame pixels → scale = 1.
        img_info = [h, w]
        img_size = (h, w)

        for d in detections:
            d["track_id"] = None

        for cid in self.class_ids:
            idxs = [i for i, d in enumerate(detections) if d["class_id"] == cid]
            cls_dets = [detections[i] for i in idxs]

            if cls_dets:
                arr = np.asarray(
                    [
                        [
                            float(d["box"][0]),
                            float(d["box"][1]),
                            float(d["box"][2]),
                            float(d["box"][3]),
                            float(d["score"]),
                        ]
                        for d in cls_dets
                    ],
                    dtype=np.float32,
                )
            else:
                arr = np.empty((0, 5), dtype=np.float32)

            if self.with_reid:
                id_feature = self._features_for(cls_dets)
                tracks = self._trackers[cid].update(
                    arr, img_info, img_size, id_feature=id_feature
                )
            else:
                tracks = self._trackers[cid].update(arr, img_info, img_size)

            assigned = _match_tracks_to_dets(
                cls_dets, tracks, iou_thr=self.match_iou, soft_iou_thr=0.01
            )
            for local_i, tid in enumerate(assigned):
                detections[idxs[local_i]]["track_id"] = tid

        self._stabilize_ids(detections)
        return detections

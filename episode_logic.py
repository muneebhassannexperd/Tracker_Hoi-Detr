"""
episode_logic.py
------------------
HOI-DETR-native pickup/putback confirmation. One persistent belief ("Episode")
per physical object, evaluated over a sliding evidence window with a refractory
period on transitions -- not a per-frame threshold with a hard reset, which is
the confirmed root cause of two real bugs found this project:
  - event flicker: one continuous hold logged as multiple pickup/putback pairs
    (found in event/hoi_event_manager.py; same root cause -- no debounce on the
    raw motion signal -- confirmed present in event/event_manager.py too)
  - phantom putback with no matching pickup, from a track-ID reassignment that
    resets state to blank (multicam/global_registry.py's
    _promote_secondary_sibling_observation)

This module is single-camera for this first validation pass: it operates on
one camera's (already appearance-stitched) OC-SORT object-track history plus
HOI-DETR's own hf-link confidence for that object, matching how the two real
bug cases we're replaying were originally logged (per-camera events). Two-
camera fusion is a follow-on step once this is validated single-camera.

Input format expected:
  hoi_json: the standard HOI-DETR JSON (frames[i].detections/hf, per
            predictions_io.py's schema)
  track_json: the STITCHED OC-SORT track history (track_history + spans),
              i.e. the output of stitch_tracks_by_appearance.py -- object
              track-ID fragmentation from fast motion is already resolved
              upstream of this module, so an episode maps 1:1 to a stitched
              object track ID for this pass.
"""
import json
import sys

# -- Tunable parameters (defaults; validate/adjust against real replay cases) --
EVIDENCE_WINDOW = 15          # frames (~0.5s @ 30fps) sliding window for net displacement
REFRACTORY_FRAMES = 45        # min frames since last confirmed transition before the
                               # opposite transition can even be evaluated
MIN_HOLD_FRAMES = 30          # min frames in HELD before a putback can be confirmed at all
DWELL_SETTLE_FRAMES = 10      # frames of near-zero net motion to confirm "settled"
HF_CONTACT_FLOOR = 0.4        # matches production's hf_confidence floor
OUTWARD_NET_PX = 40           # net displacement (px) over EVIDENCE_WINDOW to confirm pickup
INWARD_NET_PX = 40            # net displacement (px) over EVIDENCE_WINDOW to confirm putback
NOISE_PX = 6                  # per-frame jitter floor; net motion below this over the
                               # settle window counts as "stationary"

RESTING, TRANSITIONING, HELD = "RESTING", "TRANSITIONING", "HELD"


class Episode:
    def __init__(self, episode_id, track_id):
        self.episode_id = episode_id
        self.track_id = track_id
        self.state = RESTING
        self.frames_since_last_transition = 10 ** 9  # unblocked at start
        self.frames_in_current_state = 0
        self.position_history = []  # list of (frame_idx, cx, cy)
        self.events = []

    def _net_vector(self, window):
        hist = self.position_history[-window:]
        if len(hist) < 2:
            return 0.0, 0.0
        (_, x0, y0) = hist[0]
        (_, x1, y1) = hist[-1]
        return x1 - x0, y1 - y0

    def update(self, frame_idx, box, hf_prob, roi_center):
        self.frames_since_last_transition += 1
        self.frames_in_current_state += 1

        if box is None:
            # No detection this frame for this track -- a gap. Per Fix 2, this
            # is not itself evidence of anything; freeze accumulators rather
            # than reset them (the direct fix for Bug 1's "any single missed
            # frame wipes all progress").
            return

        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        self.position_history.append((frame_idx, cx, cy))
        if len(self.position_history) > EVIDENCE_WINDOW * 2:
            self.position_history = self.position_history[-EVIDENCE_WINDOW * 2:]

        contact = hf_prob is not None and hf_prob >= HF_CONTACT_FLOOR
        dx, dy = self._net_vector(EVIDENCE_WINDOW)
        # outward = away from roi_center; inward = toward it
        rx, ry = cx - roi_center[0], cy - roi_center[1]
        # project net displacement onto the outward radial direction at the
        # object's current position (a simple, ROI-shape-agnostic proxy)
        r_norm = (rx ** 2 + ry ** 2) ** 0.5 or 1.0
        outward_component = (dx * rx + dy * ry) / r_norm

        if self.state == RESTING:
            if contact and outward_component > NOISE_PX:
                self._transition(TRANSITIONING, frame_idx, "contact + early outward motion")

        elif self.state == TRANSITIONING:
            if not contact and abs(outward_component) <= NOISE_PX:
                # false start: contact lost, no real progress made -- revert
                self._transition(RESTING, frame_idx, "false start: contact lost, no net motion")
            elif (self.frames_since_last_transition >= REFRACTORY_FRAMES
                  and outward_component >= OUTWARD_NET_PX):
                self._transition(HELD, frame_idx, "pickup confirmed: net outward motion past refractory")
                self.events.append({"event": "pickup", "frame": frame_idx, "track_id": self.track_id})

        elif self.state == HELD:
            if self.frames_in_current_state < MIN_HOLD_FRAMES:
                return  # too soon to even evaluate putback -- this is the direct
                        # fix for the flicker bug: a grip adjustment right after
                        # pickup cannot immediately read as a putback
            if outward_component <= -INWARD_NET_PX:
                settled = self._is_settled()
                if settled and self.frames_since_last_transition >= REFRACTORY_FRAMES:
                    self._transition(RESTING, frame_idx, "putback confirmed: net inward motion + settled")
                    self.events.append({"event": "putback", "frame": frame_idx, "track_id": self.track_id})

    def _is_settled(self):
        hist = self.position_history[-DWELL_SETTLE_FRAMES:]
        if len(hist) < DWELL_SETTLE_FRAMES:
            return False
        xs = [p[1] for p in hist]
        ys = [p[2] for p in hist]
        return (max(xs) - min(xs) <= NOISE_PX * 2) and (max(ys) - min(ys) <= NOISE_PX * 2)

    def _transition(self, new_state, frame_idx, reason):
        self.state = new_state
        self.frames_since_last_transition = 0
        self.frames_in_current_state = 0


def load_object_track_history(track_json_path):
    trk = json.load(open(track_json_path))
    track_history = {int(k): v for k, v in trk["track_history"].items()}
    by_track = {}
    for fr, entries in track_history.items():
        for e in entries:
            if e["cls"] != 1:
                continue
            by_track.setdefault(e["tid"], {})[fr] = e["box"]
    return by_track


def load_hf_confidence_by_box(hoi_json_path):
    """{frame_idx: [(box, prob), ...]} for firstobject boxes with an hf link."""
    hoi = json.load(open(hoi_json_path))
    out = {}
    for f in hoi["frames"]:
        fr = f["frame_idx"]
        dets = f["detections"]
        rows = []
        for link in f.get("hf", []):
            b_idx = link["b"]
            if dets[b_idx]["class_name"] != "firstobject":
                continue
            rows.append((dets[b_idx]["box"], link["prob"]))
        if rows:
            out[fr] = rows
    return out, hoi["width"], hoi["height"]


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    a2 = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def run(hoi_json_path, track_json_path):
    by_track = load_object_track_history(track_json_path)
    hf_by_frame, width, height = load_hf_confidence_by_box(hoi_json_path)
    roi_center = (width / 2.0, height / 2.0)  # simple proxy: frame center

    all_events = []
    for tid, frames in by_track.items():
        ep = Episode(episode_id=tid, track_id=tid)
        min_fr, max_fr = min(frames), max(frames)
        for fr in range(min_fr, max_fr + 1):
            box = frames.get(fr)
            hf_prob = None
            if box is not None:
                candidates = hf_by_frame.get(fr, [])
                best_iou, best_prob = 0.0, None
                for cbox, prob in candidates:
                    v = _iou(box, cbox)
                    if v > best_iou:
                        best_iou, best_prob = v, prob
                if best_iou >= 0.3:
                    hf_prob = best_prob
            ep.update(fr, box, hf_prob, roi_center)
        all_events.extend(ep.events)

    all_events.sort(key=lambda e: e["frame"])
    return all_events


if __name__ == "__main__":
    events = run(sys.argv[1], sys.argv[2])
    for e in events:
        print(e)
    print(f"\ntotal: {sum(1 for e in events if e['event']=='pickup')} pickups, "
          f"{sum(1 for e in events if e['event']=='putback')} putbacks")

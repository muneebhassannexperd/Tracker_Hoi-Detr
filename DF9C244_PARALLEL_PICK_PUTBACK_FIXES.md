# df9c244 Parallel Pick / Putback Issues — Diagnosis & Fixes

Session: `df9c244e-0969-48ed-90a2-a502b39d2faa`  
Product: Barebells Cookies & Cream (qty 2 kept in `user_actvities.json`)  
Pipeline: `run_tracker_putback_v2.py` → `putback_engine_v2.py` + `standalone_single_camera_tracker.py`

**Target event counts (from labeled behavior):** 4 pickups + 2 putbacks  
(pick both → put both back → pick both again → keep)

---

## Observed sequence (from shared frames)

1. Two products in two hands; often only **one** `firstobject` box.
2. Later both products tracked → **2 pickups**. Shallow carry (cyan↔green), then put back — hands/products tracked but **putbacks missing / wrong**.
3. Customer reaches in to **pick again** (no product visible) → banner **`PUTBACK CONFIRMED`** (false putback).

Three main issues (A / B / C) explained that behavior. A related follow-up (episode track reassign) fixed wrong `object_track_id` on parallel pickups.

---

## Issue A — Two products, often only one tracked

### Problem
HOI JSON often had **two** `firstobject` detections for the two Barebells, but after `HOIJsonDetector.detect()` only **one** survived.

### Root cause
`dedupe_same_frame_detections` merged FO boxes if centroids were closer than **`FIRSTOBJECT_DEDUP_DIST_PX = 250`**.  
Measured pair: ~**86 px** apart, IoU ~**0.14** → treated as a duplicate of one item.

### What it caused
- Parallel pick could not get a stable second object track.
- Often only **1 pickup** instead of 2.
- Downstream putback/re-pick logic saw a broken “one item” session.

### Fix
Switch FO dedupe from centroid distance to **IoU > 0.5** (same idea as hand dedupe). True double-detects overlap heavily; adjacent parallel products do not.

| Item | Location |
|------|----------|
| File | `standalone_single_camera_tracker.py` |
| Constant | `FIRSTOBJECT_DEDUP_IOU` — **lines 654–664** |
| Logic | `dedupe_same_frame_detections` — **lines 667–688** |

Called from `HOIJsonDetector.detect()` (same file).

---

## Issue B — False putback on empty / re-pick

### Problem
With **no product** in frame (re-reach into shelf), system logged **`PUTBACK CONFIRMED hand=…`**.

### Root cause
`PutbackDetectorV2` allowed a **soft place** when:
- hand was deep at shelf, and  
- either `product_contact_seen` **or** enough frames since last pickup (`PUTBACK_SOFT_PLACE_MIN_AFTER_PICK`).

Empty-hand re-entry looked like “place then retreat” → false putback (`object=None`, `place_retreat` / similar).

### What it caused
- False putbacks during a real pickup attempt.
- Latched put-phase / bad event history for the rest of the session (feeds Issue C).

### Fix
1. Soft place stamp requires **`product_contact_seen`** (no time-only bypass).  
2. All putback **confirm** paths require **`product_contact_seen`** this put session.  
   Occlusion still OK via remembered-product branch after a real FO was seen.

| Item | Location |
|------|----------|
| File | `putback_engine_v2.py` |
| Notes / soft-place policy | **lines 351–354** |
| Soft place gate | **lines 990–1012** |
| Confirm requires product | **lines 1125–1132** (and following release reasons) |

---

## Issue C — Re-pick blocked after putback (`_put_phase`)

### Problem
After any putback, **`_put_phase = True` forever**, and `_should_block_new_pickup` refused **all** later pickups.

### Root cause
Session model treated “first putback” as end of pick phase for the whole video, to suppress shelf noise during place cascades — too aggressive for put-then-re-pick.

### What it caused
- Real second-cycle pickups never confirmed.
- Re-pick motion often scored as another putback instead (with Issue B).
- Could not reach **4 pickups / 2 putbacks**.

### Fix
1. Put-phase only blocks for **`PUT_PHASE_BLOCK_FRAMES = 45`** after last putback, then clears.  
2. A **confirmed pickup** clears `_put_phase` early.

| Item | Location |
|------|----------|
| File | `putback_engine_v2.py` |
| Constant | `PUT_PHASE_BLOCK_FRAMES` — **lines 361–363** |
| Timed clear in `_should_block_new_pickup` | **lines 405–422** |
| Clear on `_confirm_pickup` | **lines 459–464** |

---

## Related fix — Episode track reassignment (parallel `object_track_id`)

Not one of the original “three,” but required so Issue A’s two tracks stay two **episodes**.

### Problem
After A, tracker had FO tracks **2** and **4**, but `_reassign_matching_episode` ping-ponged one HELD episode between them (`gap=0`, dist ~86–110 &lt; `REASSIGN_PROXIMITY_PX=120`). Both pickups could log `object_track_id=2`.

### Fix
- Do **not** reassign if the episode’s current FO track is **still live** this frame.  
- Require **`gap >= REASSIGN_MIN_GAP_FRAMES` (3)**.

| Item | Location |
|------|----------|
| File | `standalone_single_camera_tracker.py` |
| `REASSIGN_MIN_GAP_FRAMES` | **line 844** |
| `_reassign_matching_episode` | **lines 1049–1100** (esp. live-id / min-gap checks **1078–1082**) |
| `_find_or_create_episode` | **lines 1115–1145** |
| Pass `live_fo_ids` in base `process_frame` | **lines 1769–1778** |
| Same in v2 `process_frame` | `putback_engine_v2.py` **lines 573–584** |

---

## Files touched (summary)

| File | Issues |
|------|--------|
| `standalone_single_camera_tracker.py` | **A** (FO dedupe), **reassign** (parallel episode IDs) |
| `putback_engine_v2.py` | **B** (false putback), **C** (put-phase), passes `live_fo_ids` for reassign |
| `run_tracker_putback_v2.py` | Unchanged (runner only) |
| `*_original.py` | Snapshots of pre-fix colleague code; not used by the runner |

---

## How to run

```bash
python run_tracker_putback_v2.py \
  --input Test-Data/df9c244e-0969-48ed-90a2-a502b39d2faa/media0.mp4 \
  --hoi-json Output/df9c244e-0969-48ed-90a2-a502b39d2faa/media0.json \
  --roi roi_config.json \
  --output outputs/df9c244e-0969-48ed-90a2-a502b39d2faa/media0_events.mp4
```

Use `media4` the same way for the other camera.

---

## Status vs target (4 pickups / 2 putbacks)

| Fix | Intent |
|-----|--------|
| A | Both products can be detected/tracked |
| B | No empty-hand false putback on re-reach |
| C | Re-pick after putback can count as new pickup(s) |
| Reassign | Parallel pickups get distinct `object_track_id`s |

Event counts on df9c244 may still need tuning (e.g. putback completeness on shallow put-backs) so both cameras consistently hit **4 / 2**; A–C address the root failure modes from the shared frames.

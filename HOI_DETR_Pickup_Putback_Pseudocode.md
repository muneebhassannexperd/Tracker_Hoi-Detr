# Pickup/Putback Logic — Pseudocode (Single Camera)

## Why it's built this way

Two design decisions run through everything below, both backed by evidence
gathered earlier rather than assumption:

1. **The hand is the primary signal; the product is a bonus confirmation
   whenever it's available.** HOI-DETR's raw detections are simply better for
   the hand — larger, more consistently shaped, less prone to dropping out —
   than for the product, which is smaller and gets partially swallowed by the
   hand holding it. We already measured this asymmetry directly: when the
   tracker was tuned on real footage, the `hand` class needed no adjustment at
   all, while the `firstobject` class needed its matching threshold loosened
   significantly to stop fragmenting.

2. **The hand-to-product link is re-verified every frame, never assumed to
   persist.** The hand goes through the *same* generic tracker as the product
   — same ID-assignment code, same possibility of a forced split or a blank
   reset. Structurally, that risk exists equally for both. In practice, the
   hand's track breaks far less often, precisely because it's rarely fed the
   kind of messy, inconsistent input that triggers those failure modes for
   the product. But "rarely" isn't "never," so the link is cheap insurance
   that gets rechecked every qualifying frame rather than trusted to hold.

3. **Hand gesture (open vs. gripping) is a third, independent evidence
   source, not a separate decision path.** Position and motion tell you
   *where* the hand is going; gesture tells you *what the hand's fingers are
   doing* — a genuinely different kind of signal. A closed, curled hand
   shape while moving away from the ROI is much stronger proof of an actual
   grasp than motion alone (a hand can drift past an item without ever
   closing on it). And a hand visibly opening back up is a more direct sign
   of release than inferring it purely from two paths splitting apart
   geometrically. This feeds into the same weighted evidence score as
   everything else, rather than becoming a fourth independent trigger that
   could fire on its own — consistent with the whole design's rule of one
   fused signal per episode, never several that could each act alone.

This version is **single camera** — everything below reads from one video
stream only, using the ROI/opening layout the way it was originally sketched
(machine on one side, customer on the other, ROI boundary in between).

## Data structures

```
Episode:
    episode_id
    state                    # RESTING | TRANSITIONING | HELD
    linked_hand_id           # refreshed every contact frame -- never assumed to persist
    contact_streak           # consecutive frames of qualifying contact
    contact_start_frame
    last_transition_frame
    hand_position_history    # short rolling window, reliable, checked every frame
    object_position_history  # sparse -- only filled in on the frames it's actually detected
    grip_history             # recent open/closed gesture readings for the linked hand
    outward_score            # distance-past-ROI-boundary, running, not frame-count-based
    frames_since_seen        # for the "lost sight of it" case
    product_name             # filled in AFTER pickup is confirmed, not before

pickups = []          # {episode_id, frame, hand_id, product_name}
putbacks = []         # {episode_id, frame}
open_episodes = {}    # episode_id -> Episode, currently TRANSITIONING or HELD
```

One record per event, holding the ID and everything about it together —
rather than parallel ID-only lists that have to be kept in sync separately.

## Main loop

Runs once per frame, single camera stream.

```
for frame_idx, frame_data in video_frames:

    hands    = detect_hands(frame_data)      # from HOI-DETR, this camera
    objects  = detect_objects(frame_data)
    hf_links = get_hf_links(frame_data)      # pairwise hand<->object relationships;
                                              # there can be MORE THAN ONE per frame

    for (hand_id, object_id, link_conf) in hf_links:

        episode = find_episode_by_object_id(object_id, open_episodes)
        if episode is None:
            episode = new_episode(object_id)          # not already claimed -> fresh episode

        episode.linked_hand_id = hand_id              # ALWAYS overwritten, never inherited

        update_contact_signal(episode, link_conf, frame_idx)
        update_motion_evidence(episode, get_position(hand_id), get_position(object_id), frame_idx)
        update_gesture_evidence(episode, read_hand_gesture(hand_id, frame_data), frame_idx)
        evaluate_transition(episode, frame_idx)

    for episode in open_episodes.values():
        if episode was not touched by any hf_link this frame:
            handle_missing_evidence(episode, hands, objects, frame_idx)
```

`find_episode_by_object_id` gates purely on "is this object already claimed
by an open episode," never on appearance — which is what keeps identical-
looking products from ever getting confused with each other. The hand link
is reassigned every single frame for the reason above: since the hand's ID
can glitch too, nothing here assumes "it was this hand last time."

## Contact — must hold up, not fire on one frame

```
def update_contact_signal(episode, link_conf, frame_idx):
    if link_conf >= CONTACT_FLOOR:
        episode.contact_streak += 1
    else:
        episode.contact_streak = 0

    if episode.state == RESTING and episode.contact_streak >= K1:
        episode.state = TRANSITIONING
        episode.contact_start_frame = frame_idx
```

`K1` (a small number of frames, not one) is the direct fix for confidence
flicker — HOI-DETR's own touch-confidence naturally wobbles even during a
real, continuous touch.

## Motion — ROI-boundary-based, hand is primary, object is a bonus

This is the part tied to the original sketch: `Machine | ROI | Customer`,
with the ROI line as the boundary — crossing toward the customer side is
outward (pickup direction), crossing back toward the machine side is inward
(putback direction).

```
def update_motion_evidence(episode, hand_pos, object_pos, frame_idx):
    episode.hand_position_history.append((frame_idx, hand_pos))

    # distance PAST the ROI boundary, signed: positive = customer side (out),
    # negative/zero = still on the machine side (in)
    depth_now  = signed_distance_past_roi(hand_pos)
    depth_prev = signed_distance_past_roi(episode.hand_position_history[-WINDOW])
    hand_delta = depth_now - depth_prev

    weight = BASE_WEIGHT
    if object_pos is not None:
        episode.object_position_history.append((frame_idx, object_pos))
        if hand_object_offset_is_stable(episode):     # real rigid coupling, not coincidence
            weight += COUPLING_BONUS

    if episode.grip_history and episode.grip_history[-1] == CLOSED:
        weight += GESTURE_BONUS                        # hand shape itself backs up a real grasp

    episode.outward_score += hand_delta * weight       # DISTANCE past the ROI line, not frame count
```

Hand-led, object-as-bonus, for the reason in the intro. Using signed
distance past the ROI line (not just "which side is it on this frame," and
not a frame-count) is what fixes fast pickups: a quick, clean grab crosses a
lot of ground in very few frames, so it scores just as well as a slow one
instead of being penalized for having fewer frames to accumulate in. The
gesture bonus works the same way as the coupling bonus — it only ever adds
confidence when it's there, it's never required on its own, so a case where
gesture can't be read (bad angle, motion blur on the fingers) doesn't block
anything.

## Hand gesture — reading open vs. closed as its own signal

```
def update_gesture_evidence(episode, grip_reading, frame_idx):
    # grip_reading is OPEN, CLOSED, or UNKNOWN for this frame -- from a
    # lightweight classifier on the hand crop, or a simple heuristic like
    # hand-box compactness/aspect-ratio versus its own resting shape
    episode.grip_history.append((frame_idx, grip_reading))
```

This is deliberately a thin function — all the actual *use* of gesture
happens where it's consumed (the bonus in `update_motion_evidence`, and the
release check below), not here. Gesture is read every frame regardless of
state, same as position, so there's always a recent reading available
whenever the rest of the logic needs one.

## State transitions

```
def evaluate_transition(episode, frame_idx):
    since_contact = frame_idx - episode.contact_start_frame
    required_bar  = graduated_bar(since_contact)         # high if claimed instantly, relaxes over time
    enough_gap    = (frame_idx - episode.last_transition_frame) >= MIN_GAP

    if episode.state == TRANSITIONING:
        if episode.outward_score >= required_bar and enough_gap:
            confirm_pickup(episode, frame_idx)
        elif episode.contact_streak == 0 and episode.outward_score < NOISE_FLOOR:
            episode.state = RESTING                      # false start -- quietly revert

    elif episode.state == HELD:
        if crossed_back_toward_machine_side(episode) and then_settles_and_separates(episode) and enough_gap:
            confirm_putback(episode, frame_idx)
```

"Settles AND separates," not just settles, for putback: settling alone can
just mean the customer paused while still holding it near the ROI — the
hand's path continuing on *after* the item stops is what actually proves
release.

`then_settles_and_separates` itself leans on gesture wherever it's
available, since an open hand is a more direct sign of release than
inferring it from motion alone:

```
def then_settles_and_separates(episode):
    settled = hand_has_stopped_moving(episode)
    geometric_separation = hand_position_diverging_from_object(episode)
    gesture_opened = recent_grip_transitioned_to(episode, OPEN)

    return settled and (geometric_separation or gesture_opened)
```

Either signal is enough once settling is confirmed — they're not stacked as
two more hurdles to clear, they're two independent ways of proving the same
thing (the hand let go), so a bad angle on one doesn't block a confirmation
the other can already support.

## Confirming events

```
def confirm_pickup(episode, frame_idx):
    episode.state = HELD
    episode.last_transition_frame = frame_idx
    episode.outward_score = 0
    pickups.append({episode: episode.episode_id, frame: frame_idx, hand: episode.linked_hand_id})
    # product identity is intentionally NOT resolved here -- see below

def confirm_putback(episode, frame_idx):
    matched = resolve_ambiguous_return(episode, open_episodes, frame_idx)
    if matched is None:
        return   # still ambiguous, wait -- see resolve_ambiguous_return
    matched.state = RESTING
    putbacks.append({episode: matched.episode_id, frame: frame_idx})
    del open_episodes[matched.episode_id]
```

## Lost sight of it — expected, not an error

Single camera, so there's no second view to fall back on — a gap here just
means the hand or product briefly wasn't detected (blur, a bad angle for a
moment, mid-grasp occlusion). That's normal, not a failure.

```
def handle_missing_evidence(episode, hands, objects, frame_idx):
    if episode.linked_hand_id not in hands and episode.object_id not in objects:
        episode.frames_since_seen += 1                    # freeze -- do not decay, do not reset
        if episode.frames_since_seen > SESSION_END_THRESHOLD:
            resolve_as_kept(episode)                       # a real, defined ending, not a stall
    else:
        episode.frames_since_seen = 0
```

Freeze instead of reset: whatever evidence was already built up stays put
while the hand is briefly gone, and picks back up exactly where it left off
the moment it's visible again, rather than throwing away real progress
because of a normal, temporary detection gap.

## Ambiguous returns — score candidates, but don't stall forever

```
def resolve_ambiguous_return(returning_episode, open_episodes, frame_idx):
    candidates = [e for e in open_episodes.values()
                  if e.state == HELD and same_category(e, returning_episode)]
    if len(candidates) <= 1:
        return candidates[0] if candidates else None

    scored = sorted(candidates, key=lambda e: -score(e, returning_episode, frame_idx))

    if scored[0].score - scored[1].score >= MARGIN_THRESHOLD:
        return scored[0]

    returning_episode.pending_resolution_frames += 1
    if returning_episode.pending_resolution_frames >= MAX_WAIT:
        mark_low_confidence(scored[0])
        return scored[0]                                   # forced resolution, flagged
    return None                                             # still waiting

def score(episode, returning_episode, frame_idx):
    return (W1 * category_match(episode, returning_episode)
          + W2 * recency(episode, frame_idx)
          + W3 * proximity(episode, returning_episode))
```

## Product identity — resolved after the event, not before

```
def on_pickup_confirmed(episode, video_frames):
    crops = extract_crops(episode.object_position_history, video_frames)   # the crop-extraction script
    episode.product_name = identify_product(crops)
    overlay_on_video(episode.product_name, episode.episode_id)
```

This stays last on purpose: the pickup/putback decision is made purely from
motion and contact, and only *then* does the system ask "which product was
it," using the crops. Event confirmation never waits on, or depends on,
identity resolution.

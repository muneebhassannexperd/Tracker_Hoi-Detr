"""
Tracker factory.

Usage:
    from hoi_trackers import build_tracker
    tracker = build_tracker('hybrid_sort_reid', det_thresh=0.3)
    tracker = build_tracker('hybrid_sort')      # no appearance
    tracker = build_tracker(None)               # disabled
"""

from typing import Optional

from .base import BaseTracker


class NoOpTracker(BaseTracker):
    """Passthrough tracker that leaves detections untracked."""

    name = "none"

    def reset(self) -> None:
        return None

    def update(self, detections, frame):
        for d in detections:
            d.setdefault("track_id", None)
        return detections


def build_tracker(name: Optional[str] = "hybrid_sort_reid", **kwargs) -> BaseTracker:
    """
    Build a tracker by name.

    Supported:
        None / 'none' / 'off'              -> NoOpTracker
        'hybrid_sort'                      -> Hybrid-SORT (TCM, no ReID)
        'hybrid_sort_reid' / 'deep_hybrid_sort' / 'hybrid_sort_deep'
                                           -> Hybrid-SORT-ReID (Deep Hybrid SORT)
    """
    if name is None:
        return NoOpTracker()
    key = str(name).strip().lower()
    if key in ("", "none", "off", "disabled"):
        return NoOpTracker()

    from .hybrid_sort import HybridSortTracker

    if key in ("hybrid_sort", "hybridsort"):
        kwargs = dict(kwargs)
        kwargs["with_reid"] = False
        return HybridSortTracker(**kwargs)

    if key in (
        "hybrid_sort_reid",
        "hybrid-sort-reid",
        "deep_hybrid_sort",
        "deep-hybrid-sort",
        "hybrid_sort_deep",
        "deep_hybridsort",
    ):
        kwargs = dict(kwargs)
        kwargs["with_reid"] = True
        return HybridSortTracker(**kwargs)

    raise ValueError(
        f"Unknown tracker {name!r}. "
        f"Supported: hybrid_sort_reid (deep), hybrid_sort, none."
    )

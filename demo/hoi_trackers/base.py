"""
Pluggable tracker interface for HOI video demos.

All trackers take per-frame HOI detections and write a `track_id` onto
each matched detection dict.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseTracker(ABC):
    """Stateful multi-object tracker over a single video."""

    name: str = "base"

    @abstractmethod
    def reset(self) -> None:
        """Clear track state (call at the start of each new video)."""

    @abstractmethod
    def update(self, detections: List[dict], frame) -> List[dict]:
        """
        Assign track IDs for the current frame.

        Parameters
        ----------
        detections : list of HOI detection dicts from run_inference
            Each has at least: box (xyxy), score, class_id, class_name.
            Optional: embedding (1D array) for appearance association.
        frame : HxWx3 BGR uint8 numpy array (original video frame)

        Returns
        -------
        The same detection list, with `track_id` (int) set when matched.
        Unmatched detections get `track_id = None`.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

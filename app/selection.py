"""Filtering tracked detections down to what the user asked to pick.

Split out from app/web.py so the filtering rule itself (what "pick a
Triangle" or "pick a red one" actually matches) is a plain function --
testable without Flask, and the one place to extend when color filtering
(grade II/III's "pick only the specified color") is added alongside shape
selection.
"""

import math
from typing import Dict, List, Optional

from vision.shapes import DetectedShape


def filter_tracked(tracked: Dict[int, DetectedShape], shape: Optional[str] = None,
                    color: Optional[str] = None) -> Dict[int, DetectedShape]:
    """Returns only the tracked detections matching shape/color. Either
    left as None (or "") matches anything -- so filter_tracked(tracked)
    with no filters returns everything, same as picking "all"."""
    return {
        object_id: detection for object_id, detection in tracked.items()
        if (not shape or detection.shape_name == shape)
        and (not color or detection.color == color)
    }


def find_same_object(original: DetectedShape, detections: List[DetectedShape]) -> Optional[DetectedShape]:
    """The detection that is most likely `original` still lying where it
    was (used to tell whether a pick actually lifted it): same shape and
    color, centroid within about half the object's size of where it was.
    Returns the closest such detection, or None."""
    reach = max(original.area, 1.0) ** 0.5 / 2  # area is the bbox area, so this is ~half its side
    best, best_distance = None, reach
    for candidate in detections:
        if candidate.shape_name != original.shape_name or candidate.color != original.color:
            continue
        distance = math.dist(candidate.pixel, original.pixel)
        if distance <= best_distance:
            best, best_distance = candidate, distance
    return best

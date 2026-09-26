"""Filtering tracked detections down to what the user asked to pick.

Split out from app/web.py so the filtering rule itself (what "pick a
Triangle" or "pick a red one" actually matches) is a plain function --
testable without Flask, and the one place to extend when color filtering
(grade II/III's "pick only the specified color") is added alongside shape
selection.
"""

from typing import Dict, Optional

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

"""Color classification for a detected object."""

from typing import Dict, List, Optional, Tuple

import numpy as np


class ColorClassifier:
    """Buckets the mean HSV hue inside an object's mask into a named color.
    The YOLO model only knows shapes, so color stays classical."""

    def __init__(self, hue_ranges: Dict[str, List[Tuple[int, int]]]):
        self._hue_ranges = hue_ranges

    def classify(self, hsv_img: np.ndarray, mask: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """Returns (color_name, mean_hue). color_name is None if no
        configured range matches; mean_hue is None if the mask is empty."""
        hue_values = hsv_img[:, :, 0][mask > 0]
        if hue_values.size == 0:
            return None, None
        mean_hue = float(np.mean(hue_values))

        for color, ranges in self._hue_ranges.items():
            if any(lo <= mean_hue <= hi for lo, hi in ranges):
                return color, mean_hue
        return None, mean_hue

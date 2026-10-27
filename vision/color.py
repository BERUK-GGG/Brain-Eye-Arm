"""Color classification for a detected object, against a fixed palette."""

from typing import Optional

import numpy as np

# The only colors the objects come in. The web UI's color dropdown lists these.
PALETTE = ("red", "green", "blue", "black", "white")

_CHANNEL_COLORS = ("blue", "green", "red")  # OpenCV's BGR channel order
MEDIAN_SAMPLE = 4000  # pixels per median; plenty for a stable color, and far cheaper than all of them


class ColorClassifier:
    """Names an object's color from PALETTE. The YOLO model only knows
    shapes, so color stays classical.

    The object's median color is divided by the table's (the white paper
    around it, i.e. the crop's pixels outside the mask). That cancels
    exposure and white balance, which differ between stations: paper
    always becomes (1, 1, 1). Then:

      brightest channel below black_max            -> black
      channels within white_max_chroma of each other -> white
      otherwise the strongest channel              -> red / green / blue

    A dominant channel rather than hue ranges, because this camera shows
    red as orange (hue ~11) and black with a green tint, which hue ranges
    got wrong. Measured on both stations: black objects reach at most
    ~0.35, colored ones at least ~0.6."""

    def __init__(self, black_max: float = 0.45, white_max_chroma: float = 0.3):
        self._black_max = black_max
        self._white_max_chroma = white_max_chroma

    def classify(self, crop_bgr: np.ndarray, mask: np.ndarray) -> Optional[str]:
        """Returns a PALETTE name, or None if the mask leaves no object or
        no table pixels to compare against."""
        inside, outside = mask > 0, mask == 0
        if not inside.any() or not outside.any():
            return None
        table = np.maximum(_median_color(crop_bgr[outside]), 1.0)
        normalised = _median_color(crop_bgr[inside]) / table

        brightest = float(normalised.max())
        if brightest < self._black_max:
            return "black"
        if (brightest - float(normalised.min())) / brightest < self._white_max_chroma:
            return "white"
        return _CHANNEL_COLORS[int(normalised.argmax())]


def _median_color(pixels: np.ndarray) -> np.ndarray:
    """Per-channel median of an (N, 3) pixel array, from an evenly spaced
    sample when N is large."""
    step = max(1, len(pixels) // MEDIAN_SAMPLE)
    return np.median(pixels[::step], axis=0).astype(np.float32)

"""Drawing of tracked detections onto a frame."""

from typing import Dict

import cv2
import numpy as np

from vision.shapes import DetectedShape

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
GRAY = (200, 200, 200)
WHITE = (255, 255, 255)


def draw_overlay(frame: np.ndarray, tracked: Dict[int, DetectedShape], scale: float = 1.0) -> np.ndarray:
    """Returns a copy of frame with each tracked detection's centroid,
    shape/color/confidence label, tracker ID, orientation angle, and the
    actual measured orientation edge drawn on it (position AND
    orientation, per the project brief's GUI requirement).

    scale: how frame relates to the camera image the detections were
    measured on -- e.g. 0.5 for a half-size frame (the web stream draws
    on a downscaled copy, which is far cheaper to encode)."""
    annotated = frame.copy()
    # Sizes below are tuned for a ~1000 px wide frame; scale them up so the
    # labels stay readable on the full-resolution (2448 px) Basler image.
    s = max(1.0, frame.shape[1] / 1000)
    for object_id, shape in tracked.items():
        u, v = int(shape.pixel[0] * scale), int(shape.pixel[1] * scale)
        cv2.circle(annotated, (u, v), int(4 * s), RED, -1)

        _text(annotated, f"ID {object_id}", (u, v - int(30 * s)), 0.6 * s, GREEN)
        _text(annotated, _label(shape), (u, max(0, v - int(8 * s))), 0.55 * s, WHITE)
        if shape.orientation_points is not None:
            _text(annotated, f"{shape.orientation:.1f} deg", (u, v + int(22 * s)), 0.55 * s, YELLOW)
            _draw_orientation_edge(annotated, shape, s, scale)
    return annotated


def _text(annotated: np.ndarray, text: str, origin, scale: float, color) -> None:
    """Text on a solid dark box, so it reads on white paper and dark parts
    alike. (A thicker black copy as an outline doesn't line up: Hershey
    glyphs get wider with stroke thickness.)"""
    thickness = max(1, int(1.5 * scale))
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = origin
    pad = max(2, int(3 * scale))
    cv2.rectangle(annotated, (x - pad, y - h - pad), (x + w + pad, y + baseline), (0, 0, 0), -1)
    cv2.putText(annotated, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_orientation_edge(annotated: np.ndarray, shape: DetectedShape, s: float = 1.0,
                           scale: float = 1.0) -> None:
    """Draws the ACTUAL two corners orientation was measured between (the
    object's current bottommost corner and next-bottommost corner), plus
    a horizontal reference line at "bottom"'s height, running out to sit
    directly above (or below) "next" -- so the angle between the two
    lines drawn is, visibly, the reported number, and the gap between
    the reference line's end and the next-bottommost corner itself is
    the vertical distance still needed to make them level. A line from
    the centroid at an abstract angle would look similar but wouldn't
    correspond to anything actually measured; this draws the real
    geometry instead."""
    bottom_point, next_point = shape.orientation_points
    bottom = (int(bottom_point[0] * scale), int(bottom_point[1] * scale))
    nxt = (int(next_point[0] * scale), int(next_point[1] * scale))

    cv2.line(annotated, bottom, (nxt[0], bottom[1]), GRAY, int(s))       # reference: level, above/below "next"
    cv2.line(annotated, bottom, nxt, YELLOW, int(2 * s))                 # measured: current bottom edge
    cv2.circle(annotated, bottom, int(4 * s), CYAN, -1)                  # bottommost corner
    cv2.circle(annotated, nxt, int(4 * s), MAGENTA, -1)                  # next-bottommost corner


def _label(shape: DetectedShape) -> str:
    label = shape.shape_name
    if shape.color:
        label = f"{shape.color}_{label.lower()}"
    if shape.confidence:
        label += f" {shape.confidence:.2f}"
    return label

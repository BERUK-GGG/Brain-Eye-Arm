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


def draw_overlay(frame: np.ndarray, tracked: Dict[int, DetectedShape]) -> np.ndarray:
    """Returns a copy of frame with each tracked detection's centroid,
    shape/color/confidence label, tracker ID, orientation angle, and the
    actual measured orientation edge drawn on it (position AND
    orientation, per the project brief's GUI requirement)."""
    annotated = frame.copy()
    for object_id, shape in tracked.items():
        u, v = int(shape.pixel[0]), int(shape.pixel[1])
        cv2.circle(annotated, (u, v), 4, RED, -1)

        cv2.putText(annotated, _label(shape), (u, max(0, v - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)
        cv2.putText(annotated, f"ID {object_id}", (u, v - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)

        if shape.orientation_points is not None:
            cv2.putText(annotated, f"{shape.orientation:.1f} deg", (u, v + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 2)
            _draw_orientation_edge(annotated, shape)
    return annotated


def _draw_orientation_edge(annotated: np.ndarray, shape: DetectedShape) -> None:
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
    bottom = (int(bottom_point[0]), int(bottom_point[1]))
    nxt = (int(next_point[0]), int(next_point[1]))

    cv2.line(annotated, bottom, (nxt[0], bottom[1]), GRAY, 1)      # reference: level, above/below "next"
    cv2.line(annotated, bottom, nxt, YELLOW, 2)                    # measured: current bottom edge
    cv2.circle(annotated, bottom, 4, CYAN, -1)                     # bottommost corner
    cv2.circle(annotated, nxt, 4, MAGENTA, -1)                     # next-bottommost corner


def _label(shape: DetectedShape) -> str:
    label = shape.shape_name
    if shape.color:
        label = f"{shape.color}_{label.lower()}"
    if shape.confidence:
        label += f" {shape.confidence:.2f}"
    return label

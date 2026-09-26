"""
Orientation for one detected object, from the crop around its YOLO
bounding box. YOLO gives shape and position but no rotation (an
axis-aligned box carries none), so it's measured here.

DEFINITION: the drop slot for each shape sits in one fixed place with
one fixed required orientation, so orientation doesn't need to be an
abstract absolute angle -- it only needs to be "how far this object is
from the pose it must be placed in." That pose is defined as: the
object's current topmost and bottommost corners are level (the same
row) -- top/bottom rather than the two topmost, because the camera's
image is upside-down relative to the physical scene, so the pair that
actually corresponds to a real top/bottom edge of the object is the
vertical extremes, not whichever two corners happen to be highest in
the (inverted) pixel frame. So:

  1. Find the shape's own corners (4 for a Square, 3 for a Triangle --
     we already know which from YOLO's class label).
  2. Take the topmost corner and the bottommost corner.
  3. The angle of the line between them, relative to horizontal, IS the
     orientation: 0 means they're already level (nothing to correct).
     Rotating the object by -orientation before placing brings it level.

A line only has meaning mod 180 degrees (flipped, it's the same line),
so this is always wrapped into (-90, 90].

Measurement carries the actual two corner points used (top_point,
bottom_point, in full-frame pixel coordinates), not just the resulting
angle -- so a caller can draw the real edge that was measured instead of
a synthetic direction indicator.

A rotationally symmetric shape (a circle) has no meaningful orientation
at all -- see shapes.py's ROTATIONALLY_SYMMETRIC, checked by the caller.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

MIN_FOREGROUND_PIXELS = 50  # below this, treat the crop as noise, not an object

Point = Tuple[float, float]


@dataclass
class Measurement:
    """Result of measuring one object.

    angle: degrees, tilt of the top two corners from level (see module
        docstring), or None if this shape has no corner finder or
        segmentation found nothing usable.
    centroid: full-frame pixel centroid (falls back to the YOLO box centre
        if segmentation fails, so position is usable even without an angle).
    top_point, bottom_point: the two corners angle was measured between, in
        full-frame pixel coordinates -- None together with angle if there
        wasn't one. For drawing the real measured edge, not a stand-in.
    mask: foreground mask of the crop, or None if segmentation failed.
    crop_box: (x1, y1, x2, y2) of that crop in the full frame, or None.
    """
    angle: Optional[float]
    centroid: Point
    top_point: Optional[Point] = None
    bottom_point: Optional[Point] = None
    mask: Optional[np.ndarray] = None
    crop_box: Optional[Tuple[int, int, int, int]] = None


def isolate_object_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """Saturation/value-based foreground segmentation of a small crop."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    mask = np.where((S > 70) | (V < 30), 255, 0).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def _largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def _square_corners(contour) -> np.ndarray:
    return cv2.boxPoints(cv2.minAreaRect(contour))  # 4x2


def _triangle_corners(contour) -> Optional[np.ndarray]:
    ok, triangle = cv2.minEnclosingTriangle(contour)
    if not ok or triangle is None:
        return None
    return triangle.reshape(-1, 2)  # 3x2


# Corner finder per shape name -- the only shapes with a meaningful
# orientation. A shape with no entry here (Circle -- rotationally
# symmetric, or anything unrecognized) simply measures no angle.
CORNER_FINDERS = {
    "Square": _square_corners,
    "Triangle": _triangle_corners,
}


def top_bottom_pair(corners: np.ndarray) -> Tuple[Point, Point]:
    """The topmost corner and the bottommost corner (each tie broken by
    picking the leftmost, so the reading doesn't flicker between frames
    when two corners share a row)."""
    pts = corners.tolist()
    top = min(pts, key=lambda p: (p[1], p[0]))       # smallest row; ties -> smallest column
    bottom = min(pts, key=lambda p: (-p[1], p[0]))   # largest row; ties -> smallest column
    return tuple(top), tuple(bottom)


def angle_between(p1: Point, p2: Point) -> float:
    """Angle (degrees) of the line p1->p2 relative to horizontal, wrapped
    into (-90, 90] since a line's orientation only means something mod 180."""
    raw = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    wrapped = raw % 180
    return wrapped - 180 if wrapped > 90 else wrapped


def tilt_from_level(corners: np.ndarray) -> float:
    """Angle (degrees, wrapped to (-90, 90]) of the line between the
    topmost corner and the bottommost corner, relative to horizontal --
    see module docstring."""
    return angle_between(*top_bottom_pair(corners))


def measure(full_frame: np.ndarray, bbox_xyxy, shape_name: str, margin: int = 10) -> Measurement:
    """
    bbox_xyxy: (x1, y1, x2, y2) full-frame pixels, from a YOLO detection.
    shape_name: which corner finder to use (see CORNER_FINDERS); a name
        with no finder measures no angle (e.g. a circle).
    margin: extra pixels around the box so the object isn't clipped.
    """
    x1, y1, x2, y2 = bbox_xyxy
    h, w = full_frame.shape[:2]
    bbox_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    cx1 = max(0, int(x1) - margin)
    cy1 = max(0, int(y1) - margin)
    cx2 = min(w, int(x2) + margin)
    cy2 = min(h, int(y2) + margin)

    if cx2 <= cx1 or cy2 <= cy1:
        return Measurement(angle=None, centroid=bbox_center)

    mask = isolate_object_mask(full_frame[cy1:cy2, cx1:cx2])

    ys, xs = np.nonzero(mask)
    if xs.size < MIN_FOREGROUND_PIXELS:
        return Measurement(angle=None, centroid=bbox_center)

    # Centroid in crop-local coordinates -> full-frame.
    centroid = (float(xs.mean() + cx1), float(ys.mean() + cy1))
    crop_box = (cx1, cy1, cx2, cy2)

    angle = top_point = bottom_point = None
    finder = CORNER_FINDERS.get(shape_name)
    if finder is not None:
        contour = _largest_contour(mask)
        if contour is not None:
            corners = finder(contour)
            if corners is not None:
                top, bottom = top_bottom_pair(corners)
                angle = angle_between(top, bottom)
                # Crop-local -> full-frame, same offset as the centroid above.
                top_point = (top[0] + cx1, top[1] + cy1)
                bottom_point = (bottom[0] + cx1, bottom[1] + cy1)

    return Measurement(angle=angle, centroid=centroid, top_point=top_point, bottom_point=bottom_point,
                        mask=mask, crop_box=crop_box)

"""
Orientation for one detected object, from the crop around its YOLO
bounding box. YOLO gives shape and position but no rotation (an
axis-aligned box carries none), so it's measured here.

DEFINITION: the drop slot for each shape sits in one fixed place with
one fixed required orientation, so orientation doesn't need to be an
abstract absolute angle -- it only needs to be "how far this object is
from the pose it must be placed in." That pose is defined as: the
object's current bottommost two corners are level (the same row) --
bottommost rather than topmost, because the camera's image is
upside-down relative to the physical scene, so the pair that actually
corresponds to the object's real top edge is its bottommost two corners
in the (inverted) pixel frame. So:

  1. Find the shape's own corners (4 for a Square, 3 for a Triangle --
     we already know which from YOLO's class label).
  2. Take the bottommost corner and the next-bottommost corner.
  3. The angle of the line between them, relative to horizontal, IS the
     orientation: 0 means they're already level (nothing to correct).
     Rotating the object by -orientation before placing brings it level.

A line only has meaning mod 180 degrees (flipped, it's the same line),
so this is always wrapped into (-90, 90].

SIGN CONVENTION -- CONFIRMED ON HARDWARE (see angle_between): a plain
image-frame angle turned out to rotate the robot the opposite way from
what the measured tilt implied. angle_between negates its result to
correct for that, so `correction = target_tilt - orientation`
(robot/picker.py) now turns the robot the right direction -- verified by
an actual pick-and-place cycle, not just assumed.

Measurement carries the actual two corner points used (bottom_point,
next_point, in full-frame pixel coordinates), not just the resulting
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

    angle: degrees, tilt of the bottom two corners from level (see module
        docstring), or None if this shape has no corner finder or
        segmentation found nothing usable.
    centroid: full-frame pixel centroid (falls back to the YOLO box centre
        if segmentation fails, so position is usable even without an angle).
    bottom_point, next_point: the two corners angle was measured between,
        in full-frame pixel coordinates -- None together with angle if
        there wasn't one. For drawing the real measured edge, not a
        stand-in.
    mask: foreground mask of the crop, or None if segmentation failed.
    crop_box: (x1, y1, x2, y2) of that crop in the full frame, or None.
    """
    angle: Optional[float]
    centroid: Point
    bottom_point: Optional[Point] = None
    next_point: Optional[Point] = None
    mask: Optional[np.ndarray] = None
    crop_box: Optional[Tuple[int, int, int, int]] = None


MIN_COLOR_DISTANCE = 25  # Lab distance floor: below this a pixel is never foreground


def isolate_object_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """Foreground = pixels that differ from the background, wherever it
    is. The background colour is taken from the crop's border (mostly
    table around the object), and the cut-off between object and table
    is picked per crop by Otsu. Unlike fixed saturation/brightness
    thresholds, this works for any object colour (black and grey
    included) and survives lighting and exposure changes."""
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[:3].reshape(-1, 3), lab[-3:].reshape(-1, 3),
                             lab[:, :3].reshape(-1, 3), lab[:, -3:].reshape(-1, 3)])
    background_colour = np.median(border, axis=0)
    distance = np.clip(_colour_distance(lab, background_colour), 0, 255).astype(np.uint8)

    otsu, _ = cv2.threshold(distance, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground = distance > max(otsu, MIN_COLOR_DISTANCE)

    # A touching neighbour of another colour also differs from the table,
    # so it would merge into one blob. Keep only pixels close to the
    # colour at the crop's centre (YOLO centres the box on the object).
    h, w = foreground.shape
    centre = (slice(h * 2 // 5, h * 3 // 5 + 1), slice(w * 2 // 5, w * 3 // 5 + 1))
    centre_pixels = lab[centre][foreground[centre]]
    if centre_pixels.size:
        object_colour = np.median(centre_pixels, axis=0)
        contrast = float(np.linalg.norm(object_colour - background_colour))
        foreground &= _colour_distance(lab, object_colour) < max(MIN_COLOR_DISTANCE, contrast / 2)
    mask = np.where(foreground, 255, 0).astype(np.uint8)

    # Opening sized to the crop wipes out thin pen lines and specks next
    # to the object; closing fills small holes (glare, print texture).
    k = max(5, min(mask.shape) // 25) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _colour_distance(lab: np.ndarray, colour: np.ndarray) -> np.ndarray:
    """Per-pixel Euclidean distance to one colour (several times faster
    than np.linalg.norm over the channel axis)."""
    diff = lab - colour
    return cv2.sqrt(cv2.transform(diff * diff, np.ones((1, 3), np.float32)))


def _object_contour(mask: np.ndarray):
    """The contour of the detected object itself: the biggest one
    containing the crop's centre (YOLO centres the box on the object),
    so tape or a neighbouring object caught in the crop can't win just
    by being larger. Falls back to the largest contour."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    centre = (mask.shape[1] / 2.0, mask.shape[0] / 2.0)
    containing = [c for c in contours if cv2.pointPolygonTest(c, centre, False) >= 0]
    return max(containing or contours, key=cv2.contourArea)


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


def bottommost_pair(corners: np.ndarray) -> Tuple[Point, Point]:
    """The bottommost corner and the next-bottommost corner (each tie
    broken by picking the leftmost, so the reading doesn't flicker
    between frames when a shape is already exactly level). See the
    module docstring for why bottommost, not topmost."""
    ordered = sorted(corners.tolist(), key=lambda p: (-p[1], p[0]))
    return tuple(ordered[0]), tuple(ordered[1])


def angle_between(p1: Point, p2: Point) -> float:
    """Angle (degrees) of the line p1->p2 relative to horizontal, wrapped
    into (-90, 90] since a line's orientation only means something mod
    180. Negated at the end -- confirmed on real hardware that the plain
    (non-negated) angle rotated the robot the wrong way; see the module
    docstring's SIGN CONVENTION note."""
    raw = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    wrapped = raw % 180
    plain = wrapped - 180 if wrapped > 90 else wrapped
    return -plain


def tilt_from_level(corners: np.ndarray) -> float:
    """Angle (degrees, wrapped to (-90, 90]) of the line between the
    bottommost corner and the next-bottommost corner, relative to
    horizontal -- see module docstring."""
    return angle_between(*bottommost_pair(corners))


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

    angle = bottom_point = next_point = None
    finder = CORNER_FINDERS.get(shape_name)
    if finder is not None:
        contour = _object_contour(mask)
        if contour is not None:
            corners = finder(contour)
            if corners is not None:
                bottom, nxt = bottommost_pair(corners)
                angle = angle_between(bottom, nxt)
                # Crop-local -> full-frame, same offset as the centroid above.
                bottom_point = (bottom[0] + cx1, bottom[1] + cy1)
                next_point = (nxt[0] + cx1, nxt[1] + cy1)

    return Measurement(angle=angle, centroid=centroid, bottom_point=bottom_point, next_point=next_point,
                        mask=mask, crop_box=crop_box)

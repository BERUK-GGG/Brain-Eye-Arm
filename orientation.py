"""
Orientation extraction for a single detected object, given the crop of
image data around its bounding box.

WHY THIS EXISTS
----------------
Your trained YOLO model outputs an axis-aligned bounding box + class
(shape, color) -- it does NOT output rotation. An axis-aligned box
carries no orientation information at all, by construction. So
orientation still comes from classical image geometry, run on the
small cropped region inside each YOLO detection instead of scanning
the whole frame for contours.

WHY THIS ISN'T PLAIN PCA (an important correction from an earlier draft)
---------------------------------------------------------------------------
The obvious first approach is PCA on the object's contour/mask points
-- find the principal axis via eigen-decomposition of the point
cloud's covariance matrix. This works for shapes with no more than
2-fold symmetry (an elongated rectangle, a scalene triangle), but it
is MATHEMATICALLY BROKEN for a shape with 3-fold or higher rotational
symmetry -- a plain equilateral triangle or a square. Testing this
directly (rotating a synthetic square through many known angles) showed
exactly why: PCA's covariance matrix is ISOTROPIC (equal eigenvalues)
for any shape with that much symmetry, so the "principal axis" it
returns is numerically arbitrary noise, not a real measurement -- the
test returned a near-constant angle regardless of the square's true
rotation.

The general, mathematically correct fix: for a shape with k-fold
rotational symmetry, its k-th order complex moment
    M_k = sum over the shape's pixels of ((x - cx) + i(y - cy))^k
rotates as M_k -> M_k * exp(i * k * theta) when the shape rotates by
theta. So angle(M_k) / k tracks the true rotation smoothly, and lands
automatically within the shape's own symmetry period (e.g. mod 90 for
k=4, a square). k=2 is exactly the ordinary PCA/elongation case (this
was verified against minAreaRect and tracks smoothly with no
discontinuities); k=3 correctly handles a triangle where minAreaRect
itself was ALSO tested and found to jump between different local-
minimum bounding boxes rather than tracking rotation continuously.
This is why compute_orientation() below takes a symmetry_order
parameter instead of assuming k=2 (plain PCA) for everything.

The specific k used for each shape is looked up from shapes.py's
DetectedShape subclasses (their SYMMETRY_ORDER), not decided here --
this module only knows how to measure an angle at a given order, not
which order any given shape should use.
"""

import cv2
import numpy as np

MIN_CONTOUR_AREA = 50  # below this, treat as noise rather than a real object


def isolate_object_mask(crop_bgr):
    """Same S/V-based foreground segmentation idea as Detector.py's
    clean_img(), scoped to a small crop instead of the whole frame.
    (clean_img()'s 'ignore left 500px' step was specific to avoiding
    the calibration marker in the full frame and doesn't apply to an
    already-cropped detection region, so it's left out here.)"""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    mask = np.where((S > 70) | (V < 30), 255, 0).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def kth_moment_angle(mask, k):
    """angle(M_k) / k in degrees, where M_k is the k-th order complex
    moment of the mask's foreground pixels about their own centroid.
    See the module docstring for why this is the right tool instead of
    plain PCA for k >= 3. Returns None if the mask has no foreground
    pixels or the resulting moment is degenerate (near-zero magnitude
    -- can happen for a close-to-perfectly-symmetric blob at certain
    exact angles; rare in practice with real camera noise)."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None

    cx, cy = xs.mean(), ys.mean()
    z = (xs - cx) + 1j * (ys - cy)
    Mk = np.sum(z ** k)

    if abs(Mk) < 1e-6:
        return None

    return float(np.degrees(np.angle(Mk)) / k)


def compute_orientation(full_frame, bbox_xyxy, symmetry_order=2, margin=10):
    """
    full_frame: the full BGR image YOLO ran inference on.
    bbox_xyxy: (x1, y1, x2, y2) in full-frame pixel coordinates, as
        given by a YOLO detection.
    symmetry_order: the shape's rotational symmetry order k (2 for a
        rectangle/asymmetric shape, 3 for an equilateral triangle, 4
        for a square, etc -- see shapes.py's SYMMETRY_ORDER per
        subclass). Pass None for a shape with no meaningful orientation
        at all (a circle) to skip the angle computation entirely.
    margin: extra pixels added around the box before cropping, so the
        object isn't clipped right at its edge (YOLO boxes are often
        a few pixels tighter than the true object outline).

    Returns (raw_angle_degrees, centroid_pixel, mask_bbox_in_full_frame).
    raw_angle_degrees is None if symmetry_order is None, or if no
    usable object pixels were found in the crop -- callers should
    treat that as "orientation unavailable", not guess a value.
    centroid_pixel falls back to the YOLO box's own center if
    segmentation fails, so position is still usable even when angle
    isn't.
    """
    x1, y1, x2, y2 = bbox_xyxy
    h, w = full_frame.shape[:2]
    bbox_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    cx1 = max(0, int(x1) - margin)
    cy1 = max(0, int(y1) - margin)
    cx2 = min(w, int(x2) + margin)
    cy2 = min(h, int(y2) + margin)

    if cx2 <= cx1 or cy2 <= cy1:
        return None, bbox_center, None

    crop = full_frame[cy1:cy2, cx1:cx2]
    mask = isolate_object_mask(crop)

    ys, xs = np.nonzero(mask)
    if xs.size < MIN_CONTOUR_AREA:
        return None, bbox_center, None

    # Centroid in crop-local coordinates -> translate back to full-frame.
    local_cx, local_cy = xs.mean(), ys.mean()
    centroid = (float(local_cx + cx1), float(local_cy + cy1))

    if symmetry_order is None:
        return None, centroid, (cx1, cy1, cx2, cy2)

    raw_angle = kth_moment_angle(mask, symmetry_order)
    return raw_angle, centroid, (cx1, cy1, cx2, cy2)
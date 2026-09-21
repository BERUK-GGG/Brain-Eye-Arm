"""
Class hierarchy for detected objects.

Replaces the old string-based classify_shape() + plain-dict detection
approach with a small class hierarchy: each shape subclass owns its own
name and (where relevant) its pick-and-place target, instead of that
logic being scattered across if/elif chains in Detector.py and
dobot_tcp_client.py.

Usage (classical, contour-based -- used by auto_label.py):
    shape = DetectedShape.from_contour(contour, pixel=(cx, cy),
                                        robot=(rx, ry), orientation=angle,
                                        area=area)

Usage (AI-based -- used by Detector.py's YOLO pipeline):
    shape = DetectedShape.from_detection(
        shape_name="Triangle", pixel=(cx, cy), robot=(rx, ry),
        raw_orientation=angle, area=area, color="red", confidence=0.92,
    )
    # raw_orientation is automatically reduced to the shape's
    # meaningful range according to its rotational symmetry -- see
    # normalize_orientation() below.

    shape.shape_name          # "Triangle", "Circle", etc.
    shape.get_place_target()  # raises NotImplementedError if this shape
                               # has no defined place target
    shape.draw(annotated_frame)
    shape.to_dict()           # backward-compatible dict, same keys
                               # detect_objects() used to return
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Tuple
import numpy as np
import cv2


@dataclass
class DetectedShape:
    contour: np.ndarray
    pixel: Tuple[float, float]
    robot: Tuple[float, float]
    orientation: float
    area: float
    circularity: float = 0.0
    vertices: int = 0
    solidity: float = 0.0
    color: str = ""            # e.g. "red" -- set when built from a YOLO detection
    confidence: float = 0.0    # YOLO's confidence for this detection, 0 for classical

    # Overridden per subclass. ClassVar keeps these as class-level
    # attributes rather than dataclass __init__ fields. place_target is
    # NOT set here -- it's injected from config.json's "place_targets" via
    # configure_place_targets() below, so placement can be retuned without
    # touching code.
    shape_name: ClassVar[str] = "Unknown"
    place_target: ClassVar[Optional[Tuple[float, float]]] = None

    # Rotational symmetry, used both by normalize_orientation() below
    # AND by orientation.py's k-th order complex moment method (see
    # that module's docstring). One number, two views of it:
    #   SYMMETRY_ORDER (k): how many times the shape maps onto itself
    #     over a full 360-degree turn -- e.g. 4 for a square, 3 for an
    #     equilateral triangle. This is exactly the k orientation.py
    #     needs for compute_orientation()'s symmetry_order argument.
    #   SYMMETRY_PERIOD_DEG: derived as 360/k below -- the smallest
    #     rotation that maps the shape onto itself, used to wrap a raw
    #     measured angle into its physically meaningful range.
    # ROTATIONALLY_SYMMETRIC=True means the shape looks identical at
    # every angle (a circle) -- orientation is meaningless, always 0,
    # and SYMMETRY_ORDER is left None since there's no angle to measure
    # at all (orientation.py treats symmetry_order=None as "skip").
    # None (for a shape that's neither) means "no known symmetry" --
    # the raw 0-360 angle is trusted as-is.
    ROTATIONALLY_SYMMETRIC: ClassVar[bool] = False
    SYMMETRY_ORDER: ClassVar[Optional[int]] = None

    @property
    def SYMMETRY_PERIOD_DEG(self) -> Optional[float]:
        if self.SYMMETRY_ORDER:
            return 360.0 / self.SYMMETRY_ORDER
        return None

    def normalize_orientation(self, raw_angle_deg: float) -> float:
        """Reduce a raw angle to this shape's physically meaningful
        range. A raw PCA/minAreaRect angle spans 0-360 degrees, but for
        a symmetric shape many of those values describe the exact same
        physical fit -- e.g. a square rotated 10 degrees looks
        identical to one rotated 100 degrees (10 + 90). Without this,
        the same real-world placement could be reported as two very
        different-looking angles depending on which of several
        equivalent contour edges the classifier happened to latch onto,
        which would make 'correct orientation' comparisons unreliable.
        """
        if self.ROTATIONALLY_SYMMETRIC:
            return 0.0
        if self.SYMMETRY_PERIOD_DEG:
            return raw_angle_deg % self.SYMMETRY_PERIOD_DEG
        return raw_angle_deg % 360.0

    def get_place_target(self) -> Tuple[float, float]:
        """Where this shape should be placed after picking. Raises if
        this subclass hasn't defined one, so an unhandled shape fails
        loudly in the pick/place loop instead of silently doing nothing
        or crashing on a None somewhere downstream."""
        if self.place_target is None:
            raise NotImplementedError(f"No place target defined for shape '{self.shape_name}'")
        return self.place_target

    @classmethod
    def configure_place_targets(cls, place_targets: dict) -> None:
        """Injects place_target onto each registered shape subclass from
        config.json's "place_targets" (shape_name -> (x, y) or None).
        Call once at startup (Detector.__init__ does this) before any
        get_place_target() call."""
        for shape_name, shape_cls in SHAPE_NAME_TO_CLASS.items():
            shape_cls.place_target = place_targets.get(shape_name)

    def to_dict(self) -> dict:
        """Backward-compatible dict, matching the old detect_objects()
        output shape -- useful while migrating callers over."""
        return {
            "pixel": self.pixel,
            "robot": self.robot,
            "orientation": self.orientation,
            "shape": self.shape_name,
            "area": self.area,
        }

    def draw(self, annotated: np.ndarray) -> None:
        """Draw this detection's contour (if any), centroid, and label."""
        if self.contour is not None and len(self.contour) > 0:
            cv2.drawContours(annotated, [self.contour], -1, (0, 255, 0), 2)
        u, v = int(self.pixel[0]), int(self.pixel[1])
        cv2.circle(annotated, (u, v), 4, (0, 0, 255), -1)
        label = self.shape_name
        if self.color:
            label = f"{self.color}_{label.lower()}"
        if self.confidence:
            label += f" {self.confidence:.2f}"
        cv2.putText(
            annotated, label, (u, max(0, v - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        )

    @classmethod
    def from_detection(cls, shape_name, pixel, robot, raw_orientation, area,
                        color="", confidence=0.0, contour=None) -> "DetectedShape":
        """Build a DetectedShape subclass instance from an ALREADY-KNOWN
        shape name -- e.g. a YOLO detection's class label -- rather than
        classifying a contour from scratch (that's from_contour(),
        still used by auto_label.py's classical bootstrap labeling).

        raw_orientation is normalized according to the resulting
        subclass's rotational symmetry (see normalize_orientation())
        before being stored, so callers never have to think about
        symmetry themselves -- they just pass in whatever angle the
        orientation extraction step measured.
        """
        shape_cls = SHAPE_NAME_TO_CLASS.get(shape_name, UnknownShape)
        instance = shape_cls(
            contour=contour if contour is not None else np.empty((0, 1, 2), dtype=np.int32),
            pixel=pixel,
            robot=robot,
            orientation=0.0,  # set below, once normalize_orientation() is available
            area=area,
            color=color,
            confidence=confidence,
        )
        instance.orientation = instance.normalize_orientation(raw_orientation)
        return instance

    @classmethod
    def from_contour(cls, contour, pixel, robot, orientation, area) -> "DetectedShape":
        """Factory: classify the contour and instantiate the matching
        subclass. This is the only place classification logic lives --
        callers never branch on shape name themselves."""
        shape_cls, circularity, vertices, solidity = cls._classify(contour)
        return shape_cls(
            contour=contour,
            pixel=pixel,
            robot=robot,
            orientation=orientation,
            area=area,
            circularity=circularity,
            vertices=vertices,
            solidity=solidity,
        )

    @staticmethod
    def _classify(contour):
        """Same geometric classification rules as the original
        classify_shape() in Detector.py -- returns a class instead of a
        string. Looked up implicitly via the return statements below."""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area <= 0 or perimeter <= 0:
            return UnknownShape, 0.0, 0, 0.0

        circularity = 4.0 * np.pi * (area / (perimeter * perimeter))
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0

        epsilon = 0.03 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)

        rect = cv2.minAreaRect(contour)
        box_w, box_h = rect[1]
        box_area = max(1.0, float(box_w * box_h))
        extent = float(area) / box_area

        # Vertex-count-based shapes are checked BEFORE the loose
        # circularity-based Circle rule below. A regular hexagon can
        # have circularity just above 0.83 (e.g. ~0.83-0.90 depending
        # on regularity), which would otherwise trip the Circle rule
        # before its vertex count (5-6) is ever consulted -- vertex
        # count is a much sharper discriminator than circularity for
        # near-round polygons like hexagons, so it takes priority.
        if vertices == 3:
            return Triangle, circularity, vertices, solidity

        if vertices == 4:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / max(1, min(w, h))
            if 0.85 <= aspect_ratio <= 1.15 and extent > 0.75:
                return Square, circularity, vertices, solidity
            return Rectangle, circularity, vertices, solidity

        if 5 <= vertices <= 6 and solidity > 0.88:
            return Hexagon, circularity, vertices, solidity

        if circularity > 0.83 and solidity > 0.9:
            return Circle, circularity, vertices, solidity

        if solidity < 0.8 and circularity < 0.65:
            return Star, circularity, vertices, solidity

        if circularity > 0.65:
            return Rounded, circularity, vertices, solidity

        return UnknownShape, circularity, vertices, solidity


# ---- Subclasses: each declares its name and its rotational symmetry
# (used to reduce a raw measured angle to the shape's physically
# meaningful range -- see normalize_orientation()). Add a new shape by
# adding a class here -- no other file needs to change. Place targets are
# NOT declared here; they come from config.json via configure_place_targets().

class Triangle(DetectedShape):
    shape_name = "Triangle"
    # Plain equilateral triangle -- 3-fold symmetric (period 120deg).
    SYMMETRY_ORDER = 3


class Circle(DetectedShape):
    shape_name = "Circle"
    # A circle looks identical at every angle -- no meaningful
    # orientation exists to measure, so this is left uncomputed
    # entirely by the orientation step (see Detector.py) and always
    # reports 0.0.
    ROTATIONALLY_SYMMETRIC = True


class Square(DetectedShape):
    shape_name = "Square"
    # 4-fold symmetric (period 90deg).
    SYMMETRY_ORDER = 4


class Rectangle(DetectedShape):
    shape_name = "Rectangle"
    # 2-fold symmetric (opposite sides equal; period 180deg). This is
    # the ordinary PCA/elongation case -- k=2.
    SYMMETRY_ORDER = 2


class Hexagon(DetectedShape):
    shape_name = "Hexagon"
    # Assumes a regular hexagon -- 6-fold symmetric (period 60deg).
    # Adjust if yours isn't perfectly regular.
    SYMMETRY_ORDER = 6


class Star(DetectedShape):
    shape_name = "Star"
    # Assumes a regular 5-pointed star -- 5-fold symmetric (period
    # 72deg). Adjust to the actual point count if different.
    SYMMETRY_ORDER = 5


class Rounded(DetectedShape):
    shape_name = "Rounded"


class UnknownShape(DetectedShape):
    shape_name = "Unknown"


# Name -> class registry, used by from_detection() to turn a YOLO
# class label's shape name into the right subclass. Keep this in sync
# whenever a subclass is added above.
SHAPE_NAME_TO_CLASS = {
    "Triangle": Triangle,
    "Circle": Circle,
    "Square": Square,
}
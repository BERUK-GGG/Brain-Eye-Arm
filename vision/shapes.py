"""
Detected-object model. One subclass per physical shape declares its name;
subclasses register themselves by name, so adding a shape means adding a
class here and nothing else.

Orientation meaning: see vision/orientation.py's module docstring. In
short, it's the tilt (degrees) needed to bring the object's current
topmost two corners level -- since the drop slot is fixed, rotating by
-orientation before placing is exactly the correction needed. A
rotationally symmetric shape (a circle) has no meaningful orientation and
always reports 0.0.

Which registered shapes the system actually reacts to is decided by
config.json ("place_targets"), not by this file -- see vision/detector.py.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, Tuple, Type


# shape_name -> subclass, filled automatically by DetectedShape.__init_subclass__.
SHAPE_REGISTRY: Dict[str, Type["DetectedShape"]] = {}


@dataclass
class DetectedShape:
    pixel: Tuple[float, float]
    robot: Tuple[float, float]
    orientation: float          # degrees; tilt from level, see module docstring
    area: float
    color: str = ""             # "" if unresolved
    confidence: float = 0.0     # YOLO confidence
    # The two corners `orientation` was measured between (top, bottom),
    # full-frame pixel coordinates -- for drawing the real measured edge.
    # None for a shape with no meaningful orientation (a circle) or when
    # measurement failed.
    orientation_points: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None

    shape_name: ClassVar[str] = "Unknown"

    # True for a shape that looks identical at every angle (a circle) --
    # orientation is meaningless there and always 0.
    ROTATIONALLY_SYMMETRIC: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "shape_name" in cls.__dict__:
            SHAPE_REGISTRY[cls.shape_name] = cls

    @classmethod
    def normalize_orientation(cls, raw_angle_deg: float) -> float:
        """A rotationally symmetric shape has no meaningful angle -- always
        0. Everything else passes through: vision/orientation.py's
        tilt-from-level measurement is already a well-defined value."""
        return 0.0 if cls.ROTATIONALLY_SYMMETRIC else raw_angle_deg

    @classmethod
    def create(cls, shape_name: str, raw_orientation: float, **fields) -> "DetectedShape":
        """Instantiate the subclass registered for shape_name, normalizing
        raw_orientation. Raises KeyError for an unknown name."""
        shape_cls = SHAPE_REGISTRY[shape_name]
        return shape_cls(orientation=shape_cls.normalize_orientation(raw_orientation), **fields)


class Triangle(DetectedShape):
    shape_name = "Triangle"


class Circle(DetectedShape):
    shape_name = "Circle"
    ROTATIONALLY_SYMMETRIC = True


class Square(DetectedShape):
    shape_name = "Square"

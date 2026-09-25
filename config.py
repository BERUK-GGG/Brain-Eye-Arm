"""
Loads every project-tunable value from config.json (model, calibration,
place targets, robot link, color ranges), so recalibrating the camera or
moving the robot station means editing config.json, not code.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

# OpenCV hue is 0-179. Red wraps around the hue circle, hence two ranges.
DEFAULT_HUE_RANGES: Dict[str, List[Tuple[int, int]]] = {
    "red": [(0, 10), (160, 179)],
    "orange": [(11, 22)],
    "yellow": [(23, 35)],
    "green": [(36, 85)],
    "blue": [(86, 130)],
    "purple": [(131, 159)],
}


@dataclass
class RobotSettings:
    """Robot link and motion constants (see robot/client.py, robot/picker.py)."""
    host: str = "192.168.5.1"
    port: int = 6601
    timeout: float = 5.0
    home_xy: Tuple[float, float] = (250.0, -170.0)
    z_safe: float = 100.0
    z_pick: float = 17.0
    suction_do_index: int = 1
    suction_settle_seconds: float = 0.5


@dataclass
class Config:
    model_path: str
    yolo_conf_threshold: float
    orientation_margin: int
    image_points: np.ndarray
    robot_points: np.ndarray
    place_targets: Dict[str, Optional[Tuple[float, float]]]
    # Model class label -> shape name, for labels the model was trained
    # under a different name (e.g. {"Rectangle": "Square"}).
    class_aliases: Dict[str, str] = field(default_factory=dict)
    hue_ranges: Dict[str, List[Tuple[int, int]]] = field(default_factory=lambda: dict(DEFAULT_HUE_RANGES))
    robot: RobotSettings = field(default_factory=RobotSettings)

    @classmethod
    def load(cls, path: str = DEFAULT_CONFIG_PATH) -> "Config":
        with open(path, "r") as f:
            data = json.load(f)

        model_path = data["model_path"]
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(path), model_path)

        calibration = data["calibration"]
        place_targets = {
            shape_name: (tuple(coords) if coords is not None else None)
            for shape_name, coords in data.get("place_targets", {}).items()
        }

        hue_ranges = dict(DEFAULT_HUE_RANGES)
        for color, ranges in data.get("hue_ranges", {}).items():
            hue_ranges[color] = [tuple(r) for r in ranges]

        robot_data = dict(data.get("robot", {}))
        if "home_xy" in robot_data:
            robot_data["home_xy"] = tuple(robot_data["home_xy"])

        return cls(
            model_path=model_path,
            yolo_conf_threshold=data["yolo_conf_threshold"],
            orientation_margin=data["orientation_margin"],
            image_points=np.array(calibration["image_points"], dtype=np.float32),
            robot_points=np.array(calibration["robot_points"], dtype=np.float32),
            place_targets=place_targets,
            class_aliases=data.get("class_aliases", {}),
            hue_ranges=hue_ranges,
            robot=RobotSettings(**robot_data),
        )

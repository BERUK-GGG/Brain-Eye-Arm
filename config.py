"""
Loads project-tunable values (model path, calibration points, place
targets) from config.json, instead of hardcoding them in detector.py /
shapes.py -- so recalibrating the camera or moving the robot station
only means editing config.json, not code.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


@dataclass
class Config:
    model_path: str
    yolo_conf_threshold: float
    orientation_margin: int
    image_points: np.ndarray
    robot_points: np.ndarray
    place_targets: Dict[str, Optional[Tuple[float, float]]]

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

        return cls(
            model_path=model_path,
            yolo_conf_threshold=data["yolo_conf_threshold"],
            orientation_margin=data["orientation_margin"],
            image_points=np.array(calibration["image_points"], dtype=np.float32),
            robot_points=np.array(calibration["robot_points"], dtype=np.float32),
            place_targets=place_targets,
        )

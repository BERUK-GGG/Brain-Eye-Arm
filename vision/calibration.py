"""Camera pixel -> robot work-object coordinate conversion."""

from typing import Tuple

import cv2
import numpy as np


class Calibration:
    """Homography from image pixels to robot x,y, fitted once from the
    matching point lists in config.json."""

    def __init__(self, image_points: np.ndarray, robot_points: np.ndarray):
        # OpenCV's canonical point-list shape is (N, 1, 2). Passing plain
        # (N, 2) silently returned None on OpenCV 5.0, so reshape explicitly.
        src = image_points.reshape(-1, 1, 2).astype(np.float32)
        dst = robot_points.reshape(-1, 1, 2).astype(np.float32)
        count = len(src)

        # getPerspectiveTransform needs exactly 4 pairs; findHomography
        # takes 4+ (least-squares fit, more robust to jogging noise at 5+).
        if count == 4:
            self._H = cv2.getPerspectiveTransform(src, dst)
        else:
            self._H, _ = cv2.findHomography(src, dst)

        # findHomography can fail and return None without raising. Fail here,
        # loudly, rather than mid-pick on the first detection.
        if self._H is None:
            raise RuntimeError(
                f"Camera->robot calibration failed for the {count} points in config.json's "
                f"\"calibration\". Check for duplicate or collinear points, or a typo."
            )

    def pixel_to_robot(self, u: float, v: float) -> Tuple[float, float]:
        point = np.array([[[u, v]]], dtype=np.float32)
        robot = cv2.perspectiveTransform(point, self._H)
        return float(robot[0, 0, 0]), float(robot[0, 0, 1])

import os
import time
from typing import Dict, Iterable, List, Optional

import cv2

from config import Config
from vision import orientation
from vision.calibration import Calibration
from vision.color import ColorClassifier
from vision.shapes import DetectedShape, SHAPE_REGISTRY


class Detector:
    """YOLO inference -> shape / orientation / color / robot coordinates for
    one frame. Collaborators are injected, so it can be built around any
    model exposing `.predict()` and `.names`; use from_config() for the real
    thing. Build once and reuse -- loading the model is far too slow to
    repeat per frame."""

    def __init__(self, model, calibration: Calibration, color_classifier: ColorClassifier,
                 assigned_shapes: Iterable[str], class_aliases: Optional[Dict[str, str]] = None,
                 conf_threshold: float = 0.5, orientation_margin: int = 10):
        self._model = model
        self._calibration = calibration
        self._color = color_classifier
        # Only shapes in this set are reported. Anything else the model
        # finds (e.g. the two shapes not assigned to this student) is ignored.
        self._assigned_shapes = set(assigned_shapes)
        self._class_aliases = class_aliases or {}
        self._conf_threshold = conf_threshold
        self._orientation_margin = orientation_margin
        self.last_inference_seconds = 0.0

    @classmethod
    def from_config(cls, config: Config) -> "Detector":
        from ultralytics import YOLO
        import torch

        if not os.path.exists(config.model_path):
            raise FileNotFoundError(
                f"YOLO weights not found at {config.model_path}. Train a model, download "
                f"best.pt, and place it there (or update config.json's model_path)."
            )

        # Leave one core free for the camera and Flask threads. Letting torch
        # claim every core made inference ~3x slower once they competed.
        cpu_count = os.cpu_count() or 1
        torch.set_num_threads(max(1, cpu_count - 1))
        print(f"torch using {torch.get_num_threads()} CPU threads (of {cpu_count} available)")

        return cls(
            model=YOLO(config.model_path),
            calibration=Calibration(config.image_points, config.robot_points),
            color_classifier=ColorClassifier(config.hue_ranges),
            assigned_shapes=config.place_targets.keys(),
            class_aliases=config.class_aliases,
            conf_threshold=config.yolo_conf_threshold,
            orientation_margin=config.orientation_margin,
        )

    def detect(self, img) -> List[DetectedShape]:
        """Runs the full pipeline on one BGR frame."""
        start = time.perf_counter()
        boxes = self._model.predict(img, conf=self._conf_threshold, verbose=False)[0].boxes
        self.last_inference_seconds = time.perf_counter() - start
        print(f"YOLO inference: {self.last_inference_seconds * 1000:.1f} ms")

        detections = []
        for box in boxes:
            shape_name = self._resolve_shape_name(box)
            if shape_name not in self._assigned_shapes or shape_name not in SHAPE_REGISTRY:
                print(f"Ignoring detection '{shape_name}' (not an assigned shape)")
                continue

            shape = self._build_shape(img, box, shape_name)
            print(
                f"{shape.shape_name.lower()}  conf={shape.confidence:.2f}  "
                f"pixel={shape.pixel}  robot=({shape.robot[0]:.2f}, {shape.robot[1]:.2f})  "
                f"color={shape.color}  orientation={shape.orientation:.2f} deg"
            )
            detections.append(shape)
        return detections

    def _resolve_shape_name(self, box) -> str:
        raw_label = self._model.names[int(box.cls[0])]
        name = raw_label.strip().capitalize()  # tolerate case differences from the label export
        return self._class_aliases.get(name, name)

    def _build_shape(self, img, box, shape_name: str) -> DetectedShape:
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        measurement = orientation.measure(img, (x1, y1, x2, y2), shape_name, self._orientation_margin)
        # 0.0 is a placeholder, not a measurement: either this shape has no
        # corner finder (e.g. a circle) or segmentation found nothing usable.
        raw_angle = measurement.angle if measurement.angle is not None else 0.0
        orientation_points = (
            (measurement.bottom_point, measurement.next_point) if measurement.bottom_point is not None else None
        )

        return DetectedShape.create(
            shape_name,
            raw_orientation=raw_angle,
            pixel=measurement.centroid,
            robot=self._calibration.pixel_to_robot(*measurement.centroid),
            area=max(0.0, (x2 - x1) * (y2 - y1)),
            color=self._classify_color(img, measurement),
            confidence=float(box.conf[0]),
            orientation_points=orientation_points,
        )

    def _classify_color(self, img, measurement: orientation.Measurement) -> str:
        if measurement.mask is None:
            return ""
        x1, y1, x2, y2 = measurement.crop_box
        hsv_crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        color, mean_hue = self._color.classify(hsv_crop, measurement.mask)
        if color is None:
            print(f"color unresolved (mean hue {mean_hue}) -- check hue_ranges in config.json")
            return ""
        return color

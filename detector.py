import time
import cv2
import numpy as np
import os

from config import Config
from shapes import DetectedShape, SHAPE_NAME_TO_CLASS
import orientation

PATH = "test_images/test_4.png"

# Roboflow labeled the square-shaped object "Rectangle" even though it's
# physically 4-fold symmetric, not 2-fold. Using the wrong symmetry order
# breaks orientation outright (a true square fed through k=2 comes out
# degenerate, not just slightly wrong) so remap the label before it reaches
# shapes.py / orientation.py.
YOLO_CLASS_TO_SHAPE_NAME = {
    "Rectangle": "Square",
}

# ==================================================
# COLOR DETECTION
# ==================================================
# The trained YOLO model's classes are plain shape names ("Triangle",
# "Circle", "Rectangle") -- color was never trained into the model. So
# color stays classical: bucket the mean HSV hue inside an object's
# segmented mask into one of HUE_RANGES below. EDIT to match your actual
# object colors if the automatic guess is unreliable -- _measure_orientation_and_color()
# logs the mean hue whenever nothing matches.

HUE_RANGES = {
    "red": [(0, 10), (160, 179)],   # red wraps around the hue circle
    "orange": [(11, 22)],
    "yellow": [(23, 35)],
    "green": [(36, 85)],
    "blue": [(86, 130)],
    "purple": [(131, 159)],
}


def guess_color(hsv_img, mask):
    """Mean hue inside an object's mask, bucketed via HUE_RANGES. Returns
    (color_name, mean_hue), with color_name=None if no configured range
    matches -- the caller reports color="" and keeps going in that case,
    since shape/position/orientation are still valid without it."""
    hue_values = hsv_img[:, :, 0][mask > 0]
    if hue_values.size == 0:
        return None, None
    mean_hue = float(np.mean(hue_values))

    for color, ranges in HUE_RANGES.items():
        for lo, hi in ranges:
            if lo <= mean_hue <= hi:
                return color, mean_hue

    return None, mean_hue


class Detector:
    """Owns the loaded YOLO model, the camera->robot calibration, and the
    detection pipeline (inference -> shape/orientation/color -> robot
    coords). Create one instance and reuse it across frames -- loading the
    model and computing the calibration matrix are both too slow to redo
    per frame, and this is also the object each thread on the Pi (once
    split out) shares."""

    def __init__(self, config: Config):
        from ultralytics import YOLO
        import torch
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(
                f"YOLO weights not found at {config.model_path}. Train a model with "
                f"dataset/colab_train.py, download best.pt, and place it there "
                f"(or update config.json's model_path)."
            )

        # Now that stream.py runs grab_loop (continuous camera pull +
        # color conversion) and Flask's request threads concurrently with
        # this model's inference, claiming EVERY core for torch leaves
        # those other threads nothing to run on -- they end up fighting
        # torch for the same cores instead of running alongside it, which
        # is a real, measured slowdown (inference went from ~1.3s to 4+s
        # once grab_loop started competing for CPU). Leave one core free
        # for everything else instead of maxing torch out.
        cpu_count = os.cpu_count() or 1
        torch.set_num_threads(max(1, cpu_count - 1))
        print(f"torch using {torch.get_num_threads()} CPU threads (of {cpu_count} available)")

        self.config = config
        self.model = YOLO(config.model_path)
        self.H = cv2.getPerspectiveTransform(config.image_points, config.robot_points)
        DetectedShape.configure_place_targets(config.place_targets)

    def pixel_to_robot(self, u, v):
        pixel = np.array([[[u, v]]], dtype=np.float32)
        robot = cv2.perspectiveTransform(pixel, self.H)
        return float(robot[0, 0, 0]), float(robot[0, 0, 1])

    def detect(self, img):
        """Runs the full pipeline on one frame. Returns (annotated_frame, detections)."""
        boxes, inference_time = self._run_inference(img)
        print(f"YOLO inference: {inference_time * 1000:.1f} ms")

        annotated = img.copy()
        detections = []

        for box in boxes:
            shape_name, shape_class, raw_label = self._resolve_shape(box)
            if shape_class is None:
                print(f"Skipping detection with unrecognized class '{raw_label}' (no matching DetectedShape subclass)")
                continue

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox_area = max(0.0, (x2 - x1) * (y2 - y1))

            raw_angle, centroid, color = self._measure_orientation_and_color(img, (x1, y1, x2, y2), shape_class)
            shape = self._build_detected_shape(shape_name, centroid, raw_angle, color, confidence, bbox_area)

            print(
                f"{shape.shape_name.lower()}  conf={confidence:.2f}  "
                f"pixel={shape.pixel}  robot=({shape.robot[0]:.2f}, {shape.robot[1]:.2f})  "
                f"color={shape.color}  orientation={shape.orientation:.2f} deg"
            )

            self._annotate(annotated, shape, (x1, y1, x2, y2))
            detections.append(shape)

        return annotated, detections

    # ---- pipeline stages -- private, but split out so each can be moved
    # onto its own thread later without restructuring detect() itself.

    def _run_inference(self, img):
        """Returns (boxes, inference_seconds) -- callers check
        inference_seconds against the project's 3-second budget."""
        start = time.perf_counter()
        results = self.model.predict(img, conf=self.config.yolo_conf_threshold, verbose=False)[0]
        elapsed = time.perf_counter() - start
        return results.boxes, elapsed

    def _resolve_shape(self, box):
        """Translates a YOLO detection's class id into the shapes.py
        subclass to use. Returns (shape_name, shape_class, raw_label);
        shape_class is None if the label doesn't match any known shape."""
        cls_id = int(box.cls[0])
        raw_label = self.model.names[cls_id]
        shape_name = raw_label.strip().capitalize()  # tolerate case differences from Roboflow export
        shape_name = YOLO_CLASS_TO_SHAPE_NAME.get(shape_name, shape_name)
        shape_class = SHAPE_NAME_TO_CLASS.get(shape_name)
        return shape_name, shape_class, raw_label

    def _measure_orientation_and_color(self, img, bbox_xyxy, shape_class):
        """Classical geometry step, run on the small crop inside a YOLO
        box: YOLO gives shape + position but never rotation, and color was
        never baked into its classes either, so both are measured here the
        same way auto_label.py pre-labeled the training set (see
        orientation.py). Returns (raw_angle_deg, centroid_px, color)."""
        symmetry_order = None if shape_class.ROTATIONALLY_SYMMETRIC else (shape_class.SYMMETRY_ORDER or 2)

        raw_angle, centroid, mask_bbox = orientation.compute_orientation(
            img, bbox_xyxy, symmetry_order=symmetry_order, margin=self.config.orientation_margin
        )
        if raw_angle is None:
            # Either a rotationally-symmetric shape (no angle to measure) or
            # segmentation found nothing in the crop -- 0.0 is a safe
            # placeholder either way, not a real measurement.
            raw_angle = 0.0

        color = ""
        if mask_bbox is not None:
            cx1, cy1, cx2, cy2 = mask_bbox
            crop = img[cy1:cy2, cx1:cx2]
            crop_mask = orientation.isolate_object_mask(crop)
            hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            guessed_color, mean_hue = guess_color(hsv_crop, crop_mask)
            if guessed_color is None:
                print(f"color unresolved (mean hue {mean_hue}) -- check/expand HUE_RANGES in detector.py")
            else:
                color = guessed_color

        return raw_angle, centroid, color

    def _build_detected_shape(self, shape_name, centroid, raw_angle, color, confidence, bbox_area):
        robot = self.pixel_to_robot(*centroid)
        return DetectedShape.from_detection(
            shape_name=shape_name,
            pixel=centroid,
            robot=robot,
            raw_orientation=raw_angle,
            area=bbox_area,
            color=color,
            confidence=confidence,
        )

    @staticmethod
    def _annotate(annotated, shape, bbox_xyxy):
        """Draws one detection's YOLO box + shape.draw()'s centroid/label
        onto the frame (shape.draw() alone only draws a contour/dot --
        there's no contour here since this came from a bounding box, not
        findContours)."""
        x1, y1, x2, y2 = bbox_xyxy
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        shape.draw(annotated)


def resize(img):
    return cv2.resize(img, (max(1, int(img.shape[1] // 2)), max(1, int(img.shape[0] // 2))))


def load_image(image_path):
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(__file__), image_path)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return img


if __name__ == "__main__":
    img = load_image(PATH)
    config = Config.load()

    if os.path.exists(config.model_path):
        print(f"Using YOLO model: {config.model_path}")
        detector = Detector(config)
        detected_img, detections = detector.detect(img)

        cv2.imshow("Detected Objects", resize(detected_img))
        cv2.waitKey(0)
    else:
        print(f"No YOLO weights at {config.model_path}.")
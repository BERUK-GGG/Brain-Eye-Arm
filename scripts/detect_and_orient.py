"""
Standalone test: run shape detection + orientation on a single saved
image, no camera or robot involved. Prints one line per detection and
saves an annotated copy next to the input image.

Usage (from the repo root):
    python scripts/detect_and_orient.py imgs/test3.png
    python scripts/detect_and_orient.py imgs/test3.png --output out.png --conf 0.4

Uses the real pipeline (config.json -> Detector.from_config), so results
here match what stream.py's detect_loop would report for the same frame.
"""

import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `vision`/`config` imports

from app.overlay import draw_overlay
from config import Config
from vision.detector import Detector


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to the image to test (e.g. imgs/test3.png)")
    parser.add_argument("--config", default=None, help="Path to config.json (default: repo root's)")
    parser.add_argument("--output", default=None,
                         help="Where to save the annotated image (default: <image>_annotated.png)")
    parser.add_argument("--conf", type=float, default=None,
                         help="Override config.json's yolo_conf_threshold")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.image):
        sys.exit(f"Image not found: {args.image}")

    config = Config.load(args.config) if args.config else Config.load()
    if args.conf is not None:
        config.yolo_conf_threshold = args.conf

    print(f"Loading model from {config.model_path} ...")
    detector = Detector.from_config(config)

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Could not read image: {args.image}")

    start = time.perf_counter()
    detections = detector.detect(img)
    elapsed = time.perf_counter() - start

    print(f"\n{len(detections)} detection(s), {elapsed * 1000:.1f} ms total "
          f"(inference: {detector.last_inference_seconds * 1000:.1f} ms)\n")

    for shape in detections:
        print(
            f"  {shape.shape_name:<10} color={shape.color or '?':<8} "
            f"conf={shape.confidence:.2f}  pixel=({shape.pixel[0]:.0f}, {shape.pixel[1]:.0f})  "
            f"robot=({shape.robot[0]:.2f}, {shape.robot[1]:.2f})  "
            f"orientation={shape.orientation:.2f} deg"
        )
    if not detections:
        print("  (none -- check yolo_conf_threshold, or that this image has an assigned shape in it)")

    # draw_overlay expects object_id -> DetectedShape; there's no tracker
    # here, so just number them in detection order.
    annotated = draw_overlay(img, dict(enumerate(detections)))

    output_path = args.output or _default_output_path(args.image)
    cv2.imwrite(output_path, annotated)
    print(f"\nAnnotated image saved to {output_path}")


def _default_output_path(image_path: str) -> str:
    root, ext = os.path.splitext(image_path)
    return f"{root}_annotated{ext}"


if __name__ == "__main__":
    main()

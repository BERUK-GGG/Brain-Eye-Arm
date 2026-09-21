"""
Standalone pick-and-place test: grabs one live camera frame, detects
objects, and runs a full pick-and-place cycle -- useful for testing the
robot link on its own, without the rest of stream.py (Flask, the live
video feed, continuous tracking) running.

For normal operation, stream.py's "/pick" button is now the primary way
to trigger a pick-and-place cycle (see its docstring) -- it reuses the
camera connection and model stream.py already has running, and picks
whatever's currently tracked instead of doing a fresh one-shot detection.
This script still exists for testing the robot link in isolation, or
when you specifically don't want stream.py's Flask server running at all.
It shares the actual TCP protocol / pick-and-place logic with stream.py
via robot_client.py, rather than duplicating it here.

CAMERA ACCESS NOTE: this opens its own connection to the Basler camera.
Basler/pylon cameras generally only allow one open connection at a time
-- if stream.py is already running (and holding the camera open), this
script will likely fail to open it. Stop stream.py first.
"""

from camera import open_camera, grab_frame, close_camera
from config import Config
from detector import Detector
from robot_client import run_pick_and_place_cycle


def grab_live_frame():
    """Opens the Basler camera just long enough to grab one frame, then
    closes it (see the module docstring's camera-access note)."""
    camera, converter = open_camera()
    try:
        frame = grab_frame(camera, converter)
        if frame is None:
            raise RuntimeError("Camera grab failed")
        return frame
    finally:
        close_camera(camera)


def detect_targets():
    config = Config.load()
    detector = Detector(config)

    frame = grab_live_frame()
    _, detections = detector.detect(frame)
    return detections


def main():
    detections = detect_targets()
    if not detections:
        print("ERROR: No valid objects detected; robot connection not opened")
        return

    try:
        results = run_pick_and_place_cycle(detections)
        print("Results:", results)
    except OSError as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
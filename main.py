"""
Entry point: wires the camera, detector, tracker, robot link and web UI
together and serves the operator HMI: live feed, detected parts, pick
controls and pick history.

    python main.py        ->  http://<pi-ip>:5000
"""

import logging
import sys

from app.service import VisionService
from app.web import create_app
from config import Config
from robot.client import RobotClient
from robot.picker import PickAndPlaceController
from vision.camera import Camera
from vision.color import PALETTE
from vision.detector import Detector
from vision.tracker import CentroidTracker


POLLED_ROUTES = ("/detections", "/status", "/history")


def main():
    # Flush every printed line at once, even when output goes to a pipe,
    # file or service log (otherwise Python holds it back in a large buffer
    # and the detection/robot lines show up late, in bursts, or not at all).
    sys.stdout.reconfigure(line_buffering=True)

    # The HMI polls these several times a second; keep them out of the
    # request log so it still shows the requests that matter (e.g. /pick).
    logging.getLogger("werkzeug").addFilter(
        lambda record: not any(f"GET {route}" in record.getMessage() for route in POLLED_ROUTES))

    config = Config.load()

    with Camera() as camera:
        detector = Detector.from_config(config)
        # min_hits=1: an object is pickable after one detection pass.
        # Raise to 2 if false positives slip through (see vision/tracker.py).
        tracker = CentroidTracker(max_disappeared=15, max_distance=50, min_hits=1)
        service = VisionService(camera, detector, tracker)

        robot = RobotClient(config.robot)
        controller = PickAndPlaceController(robot, config.robot, config.place_targets, config.target_tilt)

        service.start()
        try:
            assigned_shapes = sorted(config.place_targets)
            known_colors = list(PALETTE)
            app = create_app(service, controller, assigned_shapes, known_colors, robot=robot)
            print("HMI ready on port 5000")
            app.run(host="0.0.0.0", port=5000, threaded=True)
        finally:
            service.stop()
            #robot.close()  # sends CLOSE, ending the script on the Dobot


if __name__ == "__main__":
    main()

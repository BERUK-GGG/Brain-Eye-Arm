"""
Entry point: wires the camera, detector, tracker, robot link and web UI
together and serves the operator HMI: live feed, detected parts, pick
controls and log.

    python main.py        ->  http://<pi-ip>:5000
"""

import logging
import sys

from app.logbuffer import LogBuffer
from app.service import VisionService
from app.web import create_app
from config import Config
from robot.client import RobotClient
from robot.picker import PickAndPlaceController
from vision.camera import Camera
from vision.color import PALETTE
from vision.detector import Detector
from vision.tracker import CentroidTracker


def main():
    # Mirror everything printed into the HMI's log panel (see app/logbuffer.py).
    logs = LogBuffer(sys.stdout)
    sys.stdout = logs
    # The page polls several times a second; don't log every request.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

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
            app = create_app(service, controller, assigned_shapes, known_colors, robot=robot, logs=logs)
            print("HMI ready on port 5000")
            app.run(host="0.0.0.0", port=5000, threaded=True)
        finally:
            service.stop()
            #robot.close()  # sends CLOSE, ending the script on the Dobot


if __name__ == "__main__":
    main()

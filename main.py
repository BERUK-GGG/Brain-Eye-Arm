"""
Entry point: wires the camera, detector, tracker, robot link and web UI
together and serves the live feed with per-shape pick buttons.

    python main.py        ->  http://<pi-ip>:5000
"""

from app.service import VisionService
from app.web import create_app
from config import Config
from robot.client import RobotClient
from robot.picker import PickAndPlaceController
from vision.camera import Camera
from vision.detector import Detector
from vision.tracker import CentroidTracker


def main():
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
            known_colors = sorted(config.hue_ranges)
            create_app(service, controller, assigned_shapes, known_colors).run(host="0.0.0.0", port=5000, threaded=True)
        finally:
            service.stop()
            robot.close()  # sends CLOSE, ending the script on the Dobot


if __name__ == "__main__":
    main()

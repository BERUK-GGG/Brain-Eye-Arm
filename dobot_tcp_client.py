"""
Standalone test: grab one camera frame, detect, run one pick-and-place
cycle. For testing the robot link without the web app.

Stop main.py first: the Basler camera allows only one open connection.
"""

from config import Config
from robot.client import RobotClient
from robot.picker import PickAndPlaceController
from vision.camera import Camera
from vision.detector import Detector


def main():
    config = Config.load()
    detector = Detector.from_config(config)

    with Camera() as camera:
        frame = camera.grab()
    if frame is None:
        print("ERROR: camera grab failed")
        return

    detections = detector.detect(frame)
    if not detections:
        print("ERROR: no valid objects detected; robot connection not opened")
        return

    robot = RobotClient(config.robot)
    controller = PickAndPlaceController(robot, config.robot, config.place_targets)
    try:
        print("Results:", controller.run_cycle(detections))
    except (OSError, RuntimeError) as e:
        print(f"ERROR: {e}")
    finally:
        robot.close()


if __name__ == "__main__":
    main()

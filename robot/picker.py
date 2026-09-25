"""The pick-and-place sequence, independent of how the robot is reached."""

import threading
import time
from typing import Callable, Dict, List, Optional, Protocol, Tuple, Union

from config import RobotSettings
from vision.shapes import DetectedShape


class Robot(Protocol):
    """What the controller needs from a robot; RobotClient satisfies it,
    and so can a test double."""
    def session(self): ...
    def move_to(self, x, y, z) -> str: ...
    def set_output(self, index: int, on: bool) -> str: ...


class PickInProgress(RuntimeError):
    """A cycle is already running; only one may drive the arm at a time."""


class PickAndPlaceController:
    def __init__(self, robot: Robot, settings: RobotSettings,
                 place_targets: Dict[str, Optional[Tuple[float, float]]]):
        self._robot = robot
        self._settings = settings
        self._place_targets = place_targets
        self._busy = threading.Lock()

    def run_cycle(self, detections: Union[Dict[int, DetectedShape], List[DetectedShape]],
                  on_placed: Optional[Callable[[int], None]] = None) -> List[dict]:
        """Picks and places each detection in turn (sorted by pixel position
        for a repeatable order) and returns one result dict per shape:
            {"object_id": 3, "shape": "Triangle", "color": "red", "status": "placed"}
        status is "placed" or "skipped_no_target" (no place target in
        config.json, so nowhere to put it down).

        detections: a dict of object_id -> DetectedShape (a tracker's
        output), or a plain list (object_id is then None throughout).
        on_placed(object_id) fires right after each shape is placed, not
        batched at the end, so the caller can stop tracking it at once.

        Raises PickInProgress if a cycle is already running. Communication
        errors propagate; the robot client drops the connection so the next
        cycle reconnects."""
        if isinstance(detections, dict):
            items = list(detections.items())
        else:
            items = [(None, shape) for shape in detections]

        if not self._busy.acquire(blocking=False):
            raise PickInProgress("A pick cycle is already in progress")
        try:
            results = []
            if not items:
                return results

            with self._robot.session():
                for object_id, shape in sorted(items, key=lambda item: item[1].pixel):
                    results.append(self._handle(object_id, shape, on_placed))
            return results
        finally:
            self._busy.release()

    def _handle(self, object_id, shape: DetectedShape, on_placed) -> dict:
        result = {"object_id": object_id, "shape": shape.shape_name, "color": shape.color}

        target = self._place_targets.get(shape.shape_name)
        if target is None:
            print(f"Skipping {shape.shape_name}: no place target in config.json")
            return {**result, "status": "skipped_no_target"}

        print(
            f"Picking {shape.shape_name} ({shape.color}) at {shape.robot} "
            f"(orientation {shape.orientation:.2f} deg -- not applied to the robot)"
        )
        self._pick_and_place(shape.robot, target)
        if on_placed is not None and object_id is not None:
            on_placed(object_id)
        return {**result, "status": "placed"}

    def _pick_and_place(self, pick_xy, place_xy) -> None:
        self._touch(pick_xy, suction_on=True)
        self._touch(place_xy, suction_on=False)
        self._robot.move_to(*self._settings.home_xy, self._settings.z_safe)

    def _touch(self, xy, suction_on: bool) -> None:
        """Descend onto xy, switch suction, wait for the vacuum to grip or
        release, then rise clear again."""
        s = self._settings
        x, y = xy
        self._robot.move_to(x, y, s.z_safe)
        self._robot.move_to(x, y, s.z_pick)
        self._robot.set_output(s.suction_do_index, suction_on)
        time.sleep(s.suction_settle_seconds)
        self._robot.move_to(x, y, s.z_safe)

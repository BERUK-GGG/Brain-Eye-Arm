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
    def move_to(self, x, y, z, r=0) -> str: ...
    def set_output(self, index: int, on: bool) -> str: ...


def wrap_to_symmetry(angle: float, symmetry_deg: float) -> float:
    """Smallest rotation equivalent to `angle` for a shape that repeats
    every symmetry_deg degrees, in (-symmetry_deg/2, symmetry_deg/2].
    E.g. a triangle (120) asked to turn 110 turns -10 instead -- it ends
    up looking the same in the slot either way."""
    half = symmetry_deg / 2
    wrapped = (angle + half) % symmetry_deg - half
    return half if wrapped == -half else wrapped


class PickInProgress(RuntimeError):
    """A cycle is already running; only one may drive the arm at a time."""


class PickAndPlaceController:
    def __init__(self, robot: Robot, settings: RobotSettings,
                 place_targets: Dict[str, Optional[Tuple[float, float]]],
                 target_tilt: Optional[Dict[str, float]] = None):
        self._robot = robot
        self._settings = settings
        self._place_targets = place_targets
        # Per-shape "already correct" tilt (degrees) -- see
        # vision/orientation.py. Defaults to 0 (level) for any shape not
        # listed. See config.py's Config.target_tilt for why this default
        # isn't yet confirmed right for every shape.
        self._target_tilt = target_tilt or {}
        self._busy = threading.Lock()

    @property
    def busy(self) -> bool:
        return self._busy.locked()

    def run_cycle(self, detections: Union[Dict[int, DetectedShape], List[DetectedShape]],
                  on_placed: Optional[Callable[[int], None]] = None,
                  on_result: Optional[Callable[[dict], None]] = None) -> List[dict]:
        """Picks and places each detection in turn (sorted by pixel position
        for a repeatable order) and returns one result dict per shape:
            {"object_id": 3, "shape": "Triangle", "color": "red", "status": "placed",
             "pick": (x, y), "orientation": 12.0, "place": (x, y), "rotation": -12.0}
        (pick/orientation always; place/rotation only when placed)
        status is "placed" or "skipped_no_target" (no place target in
        config.json, so nowhere to put it down).

        detections: a dict of object_id -> DetectedShape (a tracker's
        output), or a plain list (object_id is then None throughout).
        on_placed(object_id) fires right after each shape is placed, not
        batched at the end, so the caller can stop tracking it at once.
        on_result(result) likewise fires per shape as soon as it's handled,
        so results survive a later failure in the same cycle.

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
                    if on_result is not None:
                        on_result(results[-1])
            return results
        finally:
            self._busy.release()

    def _handle(self, object_id, shape: DetectedShape, on_placed) -> dict:
        result = {"object_id": object_id, "shape": shape.shape_name, "color": shape.color,
                  "pick": shape.robot, "orientation": shape.orientation}

        target = self._place_targets.get(shape.shape_name)
        if target is None:
            print(f"Skipping {shape.shape_name}: no place target in config.json")
            return {**result, "status": "skipped_no_target"}

        # How far to rotate the object (while still holding it) so it ends
        # up at this shape's target tilt instead of wherever it currently
        # sits. E.g. orientation=20, target=0 -> correction=-20: rotate the
        # wrist -20 degrees to bring it from 20 down to level.
        correction = wrap_to_symmetry(
            self._target_tilt.get(shape.shape_name, 0.0) - shape.orientation, shape.SYMMETRY_DEG)
        print(
            f"Picking {shape.shape_name} ({shape.color}) at {shape.robot}, orientation "
            f"{shape.orientation:.2f} deg -- rotating {correction:.2f} deg before placing"
        )
        self._pick_and_place(shape.robot, target, correction)
        if on_placed is not None and object_id is not None:
            on_placed(object_id)
        return {**result, "status": "placed", "place": target, "rotation": correction}

    def _pick_and_place(self, pick_xy, place_xy, correction: float) -> None:
        # Pick at the wrist's default (taught) rotation -- a suction cup
        # grips a flat top regardless of approach angle, so there's no need
        # to pre-align with the object; it comes up still at its own
        # as-found tilt, now rigidly attached to the gripper. Rotating by
        # `correction` on the way to the place point (while still holding
        # it) is what actually reorients the object in the world -- moving
        # the arm around at a fixed wrist angle does nothing to it.
        self._touch(pick_xy, suction_on=True, r=0)
        self._touch(place_xy, suction_on=False, r=correction)
        # Reset the wrist back to default before the next cycle's pick.
        self._robot.move_to(*self._settings.home_xy, self._settings.z_safe, 0)

    def _touch(self, xy, suction_on: bool, r: float) -> None:
        """Descend onto xy at rotation r, switch suction, wait for the
        vacuum to grip or release, then rise clear again."""
        s = self._settings
        x, y = xy
        self._robot.move_to(x, y, s.z_safe, r)
        self._robot.move_to(x, y, s.z_pick, r)
        self._robot.set_output(s.suction_do_index, suction_on)
        time.sleep(s.suction_settle_seconds)
        self._robot.move_to(x, y, s.z_safe, r)

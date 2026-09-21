"""
Lightweight centroid-based tracker to give stable IDs to detections
across frames from Detector.py's detect_objects().

WHY THIS, NOT A HEAVIER TRACKER
--------------------------------
Your scene is static objects on a table with a fixed overhead camera --
not fast-moving targets needing motion prediction (Kalman/SORT) or
appearance re-identification (DeepSORT). Nearest-centroid matching
frame-to-frame is enough to:
  - Give each object a persistent ID so you don't double-pick or lose
    track of "which one did I already send to the robot."
  - Survive a few frames of missed/occluded detection (e.g. the robot
    arm passing through view during a pick) without dropping the ID.

TUNING
------
- max_distance: how far (in pixels) a detection can move between
  frames and still be considered the "same" object. Since your
  objects are static, this mainly needs to cover camera/detection
  jitter, not real motion -- start small (e.g. 30-50px) and increase
  only if you see IDs flickering/duplicating for objects that haven't
  actually moved.
- max_disappeared: how many consecutive frames an object can go
  undetected (e.g. occluded by the robot arm) before its ID is
  dropped. Set this to cover roughly how long your arm blocks the
  view during a pick pass, at your camera's frame rate.
"""

from collections import OrderedDict
import numpy as np


class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=50):
        self.next_object_id = 0
        self.objects = OrderedDict()       # object_id -> latest detection dict
        self.disappeared = OrderedDict()   # object_id -> consecutive missed frames
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, detection):
        self.objects[self.next_object_id] = detection
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        """
        detections: list of DetectedShape instances, as produced by
        detector.py's Detector.detect() (shapes.py's DetectedShape is a
        dataclass -- attribute access, e.g. d.pixel, not d["pixel"]).
        Returns an OrderedDict of object_id -> DetectedShape (the latest
        pixel/robot/orientation/shape/color/confidence for each tracked
        object).
        """
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.array([d.pixel for d in detections], dtype=np.float32)

        if len(self.objects) == 0:
            for det in detections:
                self.register(det)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = np.array(
            [self.objects[oid].pixel for oid in object_ids], dtype=np.float32
        )

        # pairwise distances: existing tracked objects x new detections
        D = np.linalg.norm(
            object_centroids[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2
        )

        # greedy nearest-neighbor matching, closest pairs first
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue  # too far apart to be the same object

            object_id = object_ids[row]
            self.objects[object_id] = detections[col]  # refresh with latest data
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(D.shape[0])) - used_rows
        unused_cols = set(range(D.shape[1])) - used_cols

        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        for col in unused_cols:
            self.register(detections[col])

        return self.objects
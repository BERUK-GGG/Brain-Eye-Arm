"""
Centroid tracker: gives detections stable IDs across detection passes.

The scene is static objects under a fixed camera, so nearest-centroid
matching is enough -- no motion prediction or appearance matching. It
keeps an ID through a few missed passes (e.g. the arm blocking the view)
and lets a caller drop an object the instant it has been picked.

Tuning:
- max_distance: pixels a detection may move between passes and still be the
  same object. Objects are static, so this only covers detection jitter.
- max_disappeared: consecutive missed passes before an ID is dropped.
- min_hits: matched passes a NEW object needs before update() returns it.
  A one-pass YOLO false positive never reaches min_hits and ages out
  unseen. Note a "pass" here is one full inference (~1.2-1.8 s), not a
  camera frame, so min_hits=3 means up to ~5 s before an object is
  pickable. hits never resets on a miss, so a confirmed object stays
  confirmed. min_hits=1 disables the filter.

Thread-safe: update() and discard() share one internal lock, so the
detection thread and a request thread can both call in.
"""

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import List

import numpy as np

from vision.shapes import DetectedShape


@dataclass
class Track:
    shape: DetectedShape
    disappeared: int = 0    # consecutive passes without a match
    hits: int = 1           # total passes matched; never decreases


class CentroidTracker:
    def __init__(self, max_disappeared: int = 15, max_distance: float = 50, min_hits: int = 2):
        self._tracks: "OrderedDict[int, Track]" = OrderedDict()
        self._next_id = 0
        self._lock = threading.Lock()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_hits = min_hits

    def update(self, detections: List[DetectedShape]) -> "OrderedDict[int, DetectedShape]":
        """Match this pass's detections to existing tracks. Returns
        object_id -> DetectedShape for confirmed tracks only (min_hits)."""
        with self._lock:
            if not detections:
                self._mark_missed(list(self._tracks))
            elif not self._tracks:
                for det in detections:
                    self._register(det)
            else:
                self._match(detections)
            return self._confirmed()

    def discard(self, object_id: int) -> None:
        """Remove a track for a reason update() can't know about (e.g. the
        robot just picked it). Without this it would linger at its old,
        now-empty spot for many passes and could be picked again. Safe to
        call for an ID that's already gone."""
        with self._lock:
            self._tracks.pop(object_id, None)

    def _register(self, detection: DetectedShape) -> None:
        self._tracks[self._next_id] = Track(detection)
        self._next_id += 1

    def _mark_missed(self, object_ids) -> None:
        for object_id in object_ids:
            track = self._tracks[object_id]
            track.disappeared += 1
            if track.disappeared > self.max_disappeared:
                del self._tracks[object_id]

    def _confirmed(self) -> "OrderedDict[int, DetectedShape]":
        return OrderedDict(
            (oid, t.shape) for oid, t in self._tracks.items() if t.hits >= self.min_hits
        )

    def _match(self, detections: List[DetectedShape]) -> None:
        object_ids = list(self._tracks)
        object_centroids = np.array([self._tracks[oid].shape.pixel for oid in object_ids], dtype=np.float32)
        input_centroids = np.array([d.pixel for d in detections], dtype=np.float32)

        # pairwise distances: existing tracks x new detections
        D = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2)

        # greedy nearest-neighbor matching, closest pairs first
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue  # too far apart to be the same object

            track = self._tracks[object_ids[row]]
            track.shape = detections[col]  # refresh with latest data
            track.disappeared = 0
            track.hits += 1
            used_rows.add(row)
            used_cols.add(col)

        self._mark_missed([object_ids[r] for r in range(D.shape[0]) if r not in used_rows])
        for col in range(D.shape[1]):
            if col not in used_cols:
                self._register(detections[col])

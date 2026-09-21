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
- min_hits: how many total matched frames a NEW object needs before
  it's exposed to callers (i.e. before it shows up in update()'s
  return value) -- see PHANTOM DETECTIONS below.

PHANTOM DETECTIONS
-------------------
Originally, a brand-new detection was registered AND immediately
returned in the same call -- so a single-frame YOLO false positive
(a flicker, a momentary double-box on one physical object, a glare
artifact) became a fully tracked, fully "real" object the instant it
was seen once, with its own ID, sitting in the returned dict right
alongside genuine objects. Nothing told them apart -- a consumer
like stream.py's /pick route just gets a dict of DetectedShape and
has no way to know one entry was only ever seen for one frame.
That's the likely cause of "more in _latest_tracked than in real
life, so the robot goes to unnecessary locations."

The fix: new objects start TENTATIVE (hits=1) and are only included
in update()'s return value once they've accumulated min_hits total
matches. hits only ever goes up (on a match) and is never reset by a
miss -- so a real object that's confirmed once STAYS confirmed
through the odd missed frame (that's what max_disappeared is for;
see below), rather than dropping out of the returned dict and
having to re-earn confirmation every time detection briefly flickers
on a real object. A one-off spurious detection that's never matched
again simply ages out via max_disappeared, at hits=1, without ever
being returned to a caller.

NOTE ON TIMING: unlike a typical webcam tracker, update() here is
only called once per detect_loop iteration in stream.py -- i.e. once
per full YOLO inference pass (~1.2-1.8s on this hardware), NOT once
per camera frame (grab_loop runs much faster, ~15fps, but detect_loop
doesn't). So min_hits is really "how many detection passes", not
"how many camera frames" -- min_hits=3 means a real object can take
up to ~3.5-5.4s to first appear as pickable, which reads as
"detection feels slow" even though nothing is actually slow, it's
just being cautious. min_hits=2 (the default below) roughly halves
that first-appearance delay while still requiring a detection to
survive into a second, independent inference pass before being
trusted -- still filters a genuine single-pass flicker, just isn't as
conservative as 3. Raise it back toward 3 if occasional phantoms
still get through; drop to 1 (no filtering at all, the original
behavior) only if you're confident false positives aren't happening
on your setup.
"""

from collections import OrderedDict
import numpy as np


class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=50, min_hits=2):
        self.next_object_id = 0
        self.objects = OrderedDict()       # object_id -> latest detection dict
        self.disappeared = OrderedDict()   # object_id -> consecutive missed frames
        self.hits = OrderedDict()          # object_id -> total frames matched so far (never decreases)
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_hits = min_hits

    def register(self, detection):
        object_id = self.next_object_id
        self.objects[object_id] = detection
        self.disappeared[object_id] = 0
        self.hits[object_id] = 1  # tentative -- not yet confirmed, see PHANTOM DETECTIONS above
        self.next_object_id += 1
        return object_id

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.hits[object_id]

    def discard(self, object_id):
        """Like deregister(), but for an EXTERNAL caller removing an
        object that's no longer valid for a reason update() itself
        can't know about -- e.g. stream.py's /pick route, the instant a
        tracked object is actually picked up and moved by the robot.
        Without this, that object would keep sitting in update()'s
        return value at its old, now-empty pixel location until it
        naturally ages out via max_disappeared (which, at ~1-2s per
        detection pass, can be many seconds) -- long enough for a
        second pick to be sent to an already-empty spot. Safe to call
        even if object_id is already gone (a no-op then). NOT
        thread-safe on its own -- callers on a different thread from
        the one driving update() (e.g. a Flask request thread vs.
        stream.py's detect_loop thread) MUST hold the same lock around
        both update() and discard() calls; see stream.py."""
        self.objects.pop(object_id, None)
        self.disappeared.pop(object_id, None)
        self.hits.pop(object_id, None)

    def _confirmed(self):
        """Only objects matched on >= min_hits total frames -- what
        update() actually returns to callers. self.objects itself keeps
        tentative entries too, so they can still accumulate hits (or
        age out) on subsequent calls."""
        return OrderedDict(
            (oid, det) for oid, det in self.objects.items()
            if self.hits[oid] >= self.min_hits
        )

    def update(self, detections):
        """
        detections: list of DetectedShape instances, as produced by
        detector.py's Detector.detect() (shapes.py's DetectedShape is a
        dataclass -- attribute access, e.g. d.pixel, not d["pixel"]).
        Returns an OrderedDict of object_id -> DetectedShape, but ONLY
        for objects confirmed via min_hits total matches (see
        PHANTOM DETECTIONS above) -- a brand-new, not-yet-confirmed
        detection is tracked internally but withheld from this return
        value until it proves out.
        """
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self._confirmed()

        input_centroids = np.array([d.pixel for d in detections], dtype=np.float32)

        if len(self.objects) == 0:
            for det in detections:
                self.register(det)
            return self._confirmed()

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
            self.hits[object_id] += 1
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

        return self._confirmed()
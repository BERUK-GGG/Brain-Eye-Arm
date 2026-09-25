import threading
import time
from typing import Dict

from vision.camera import Camera
from vision.detector import Detector
from vision.shapes import DetectedShape
from vision.tracker import CentroidTracker


class VisionService:
    """Runs the camera grab and detection threads and holds their shared
    results. Consumers (the web layer) only read snapshots; they never
    touch the camera, detector or tracker.

    Two threads, because inference (~1.2-1.8 s) must not gate the video:
      - grab: pulls frames at a capped rate and keeps the latest.
      - detect: runs the detector on the latest frame at its own pace and
        stores the tracked detections (not an image).
    The video is drawn from the freshest frame at send time, so only the
    boxes lag inference, not the picture.

    The grab rate is capped because unthrottled color conversion competed
    with YOLO for CPU and made inference several times slower.
    """

    def __init__(self, camera: Camera, detector: Detector, tracker: CentroidTracker,
                 grab_fps: float = 15):
        self._camera = camera
        self._detector = detector
        self._tracker = tracker
        self._grab_interval = 1.0 / grab_fps

        self._lock = threading.Lock()   # guards the two fields below
        self._latest_frame = None
        self._latest_tracked: Dict[int, DetectedShape] = {}
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._grab_loop, daemon=True).start()
        threading.Thread(target=self._detect_loop, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def latest_frame(self):
        with self._lock:
            return self._latest_frame

    def tracked_snapshot(self) -> Dict[int, DetectedShape]:
        """A copy, safe to iterate while the service keeps updating."""
        with self._lock:
            return dict(self._latest_tracked)

    def discard(self, object_id: int) -> None:
        """Stop tracking an object right now (it was just picked), rather
        than waiting for it to age out and risk a second pick at the
        now-empty spot."""
        self._tracker.discard(object_id)
        with self._lock:
            self._latest_tracked.pop(object_id, None)

    def _grab_loop(self) -> None:
        while not self._stop.is_set() and self._camera.is_grabbing:
            frame = self._camera.grab()
            if frame is not None:
                with self._lock:
                    self._latest_frame = frame
            time.sleep(self._grab_interval)

    def _detect_loop(self) -> None:
        # No sleep: detector.detect() itself paces this loop.
        while not self._stop.is_set():
            frame = self._frame_copy()
            if frame is None:
                time.sleep(0.05)
                continue

            tracked = self._tracker.update(self._detector.detect(frame))
            with self._lock:
                self._latest_tracked = dict(tracked)  # snapshot: the tracker keeps changing its own

    def _frame_copy(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

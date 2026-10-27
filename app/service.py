import threading
import time
from typing import Dict, List

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

    Detection is paused while the arm works (pause()/resume()): the arm
    passing through the view hides and "moves" parts, which made the
    tracker drop them and re-register them under new IDs. While paused
    the last tracked snapshot stays frozen, and redetect() takes one-off
    looks that bypass the tracker (to check whether a pick worked).
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
        self._last_frame_time = None
        self._last_detect_time = None
        self._detect_seconds = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._detect_lock = threading.Lock()   # one detector pass at a time

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

    def stats(self) -> dict:
        """Ages (seconds) of the newest frame and detection pass, and how
        long the last pass took -- None for anything that hasn't happened yet."""
        now = time.monotonic()
        with self._lock:
            return {
                "frame_age": None if self._last_frame_time is None else now - self._last_frame_time,
                "detection_age": None if self._last_detect_time is None else now - self._last_detect_time,
                "detect_seconds": self._detect_seconds,
                "paused": self._paused.is_set(),
            }

    def pause(self) -> None:
        """Stops tracker updates; returns once no detection pass is running,
        so nothing seen after this call reaches the tracker."""
        self._paused.set()
        with self._detect_lock:
            pass

    def resume(self) -> None:
        self._paused.clear()

    def redetect(self, settle_seconds: float = 0.3, timeout: float = 5.0) -> List[DetectedShape]:
        """One detection pass on a frame grabbed at least settle_seconds
        from now (so the arm has stopped and the image isn't blurred),
        without updating the tracker. Raises RuntimeError if no such frame
        arrives within timeout."""
        wanted_after = time.monotonic() + settle_seconds
        deadline = wanted_after + timeout
        while True:
            with self._lock:
                if self._last_frame_time is not None and self._last_frame_time >= wanted_after:
                    frame = self._latest_frame.copy()
                    break
            if time.monotonic() > deadline:
                raise RuntimeError("No fresh camera frame to check the pick with")
            time.sleep(0.05)
        with self._detect_lock:
            return self._detector.detect(frame)

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
                    self._last_frame_time = time.monotonic()
            time.sleep(self._grab_interval)

    def _detect_loop(self) -> None:
        # No sleep: detector.detect() itself paces this loop.
        while not self._stop.is_set():
            frame = self._frame_copy()
            if frame is None or self._paused.is_set():
                time.sleep(0.05)
                continue

            with self._detect_lock:
                if self._paused.is_set():
                    continue  # paused while this frame was being fetched
                start = time.monotonic()
                tracked = self._tracker.update(self._detector.detect(frame))
            with self._lock:
                self._latest_tracked = dict(tracked)  # snapshot: the tracker keeps changing its own
                self._last_detect_time = time.monotonic()
                self._detect_seconds = self._last_detect_time - start

    def _frame_copy(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

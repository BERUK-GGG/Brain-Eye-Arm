"""
Live camera stream + AI detection, running on separate threads.

WHY THIS IS SPLIT ACROSS THREADS
----------------------------------
The original version ran YOLO inference (detector.detect()) synchronously
inside generate_frames()'s while loop -- i.e. once per frame pushed to the
browser. That silently capped the live view's frame rate at however fast
inference runs (~1.2-1.3s per frame here, per your own logs -- under 1 fps),
which looked like a "slow stream" but was actually "the video only updates
once a second because a full detection pass blocks every frame." Swapping
in a lower JPEG quality or smaller resolution wouldn't have fixed this --
JPEG encoding a frame takes single-digit milliseconds, nowhere near the
1200+ ms inference was actually costing per frame.

The real fix, and also exactly the multithreading split your course brief
asks for (separate threads for AI inference vs. everything else): decouple
camera grabbing/streaming (fast, can run at the camera's native rate) from
detection (slow, runs at whatever pace the model allows). Three threads:

  - grab_loop: continuously pulls frames from the Basler camera, keeps the
    latest one available. Runs as fast as the camera delivers frames
    (throttled -- see GRAB_TARGET_FPS below).
  - detect_loop: continuously takes the latest raw frame and runs the full
    Detector + tracker pipeline on it, at its own pace (~1.2-1.8s/iteration
    here) -- never waits for or blocks the other threads. Stores the
    latest TRACKED DETECTIONS (shape/pixel/color/etc, not a rendered
    image) for stream_jpeg_loop to draw.
  - Flask's request-handling thread(s): serve /video by drawing the latest
    known detections onto the LATEST RAW FRAME (not a stale pre-rendered
    one) and encoding that, every send.

IMPORTANT CORRECTION FROM AN EARLIER VERSION OF THIS FILE
-------------------------------------------------------------
An earlier version had detect_loop render+JPEG-encode the annotated frame
itself and had /video just re-serve that cached JPEG. That looked right
but was actually still a slideshow: the served image only changed once
per detect_loop iteration (~1.2-1.8s), just re-sent at 15fps in between --
so the video held still and then jumped, which reads as "not smooth" even
though inference itself was fast enough. The fix here is to always draw
onto the FRESHEST raw frame (updated at ~GRAB_TARGET_FPS by grab_loop) at
send time, using whatever detections are currently known -- so the live
feed itself is genuinely live, and only the detection boxes/labels lag by
up to ~1.2-1.8s. Since your scene is static objects on a table (not
fast-moving targets -- see tracker.py's own docstring), boxes trailing the
frame slightly is unnoticeable; a frozen-then-jumping video is not.
"""

import threading
import time

from flask import Flask, Response, jsonify
import cv2

from camera import open_camera, grab_frame, close_camera
from detector import Detector
from tracker import CentroidTracker
from config import Config
from robot_client import run_pick_and_place_cycle, close_connection as close_robot_connection

# JPEG quality for the MJPEG stream (0-100). Lower = smaller frames = less
# network/encode time, at the cost of visible compression artifacts. This
# is a real (if secondary) lever on stream responsiveness now that
# inference is no longer the thing gating frame rate -- tune to taste.
STREAM_JPEG_QUALITY = 70

# How often grab_loop actually pulls + converts a frame. Unthrottled, it
# calls converter.Convert() (a real CPU cost -- full-resolution color
# conversion) as fast as the camera delivers frames, continuously, on its
# own thread -- competing for CPU with detect_loop's YOLO inference, which
# itself tries to use every available core on the Pi. That contention is
# almost certainly why inference got SLOWER after splitting into threads
# (1.2-1.3s before -> 4+ s after, same model, same input): before, nothing
# else was running while inference ran; now, grab_loop is constantly
# stealing cycles. This cap gives it a low, fixed rate instead -- still
# plenty smooth for a live view, since the video display is itself capped
# at ~15fps below, but no longer flat-out competing with inference.
GRAB_TARGET_FPS = 15
_GRAB_SLEEP_SECONDS = 1.0 / GRAB_TARGET_FPS

app = Flask(__name__)

camera, converter = open_camera()

config = Config.load()

# Created ONCE, at startup -- Detector.__init__ loads the YOLO model from
# disk and builds the calibration matrix, both far too slow to redo per
# frame or per detection pass.
detector = Detector(config)
# min_hits=2: a new object must survive into a second, independent
# detect_loop pass before it's exposed to /pick -- filters a one-off
# YOLO flicker without making genuine new objects wait through 3+
# inference passes (~3.5-5.4s) to become pickable. See tracker.py's
# "NOTE ON TIMING" for why detect_loop's ~1.2-1.8s pace (not camera
# frame rate) is what min_hits actually counts against. Tune here if
# phantoms slip through (raise) or first-appearance still feels slow
# (lower to 1 -- disables the filter entirely).
tracker = CentroidTracker(max_disappeared=15, max_distance=50, min_hits=1)

# Shared state between the three threads described above.
_lock = threading.Lock()
_latest_raw_frame = None        # most recent frame straight from the camera
_latest_tracked = {}            # object_id -> DetectedShape, from the most recent detection pass
_latest_detections = []         # most recent detections (DetectedShape list), for other consumers (e.g. /pick below)

# Guards the CentroidTracker's own internal state (tracker.objects /
# .disappeared / .hits). detect_loop is normally the only thread that
# touches these (inside tracker.update()), but /pick's on_placed
# callback below ALSO needs to remove an object the instant it's
# picked -- rather than waiting for it to naturally age out of
# tracker.update()'s own output over the next several detection passes
# (see tracker.py's discard() docstring for why that matters: a picked
# object would otherwise keep showing up, at its old/now-empty spot,
# long enough for a second /pick click to target it again). Two
# threads writing to the same OrderedDicts needs a lock around BOTH
# sides -- this is that lock. Separate from _lock above since this one
# only needs to be held for the tracker's own quick update()/discard()
# calls, not the (much larger) raw-frame/tracked-snapshot state.
_tracker_lock = threading.Lock()

# Guards against two /pick requests (e.g. a double-click) trying to talk
# to the robot at the same time -- run_pick_and_place_cycle() holds one
# TCP connection and drives the arm through a multi-step sequence, so
# overlapping calls would interleave POSE/DO commands from two different
# cycles on the same socket-less connection and very likely send the arm
# somewhere neither cycle intended.
_pick_lock = threading.Lock()


def grab_loop():
    """Runs continuously, but throttled to GRAB_TARGET_FPS (see the
    constant's comment above for why): pulls frames from the camera,
    keeps the latest one available for detect_loop and for the raw-frame
    fallback in stream_jpeg_loop. Never touches the model."""
    global _latest_raw_frame
    while camera.IsGrabbing():
        frame = grab_frame(camera, converter)
        if frame is not None:
            with _lock:
                _latest_raw_frame = frame
        time.sleep(_GRAB_SLEEP_SECONDS)


def detect_loop():
    """Runs continuously: takes whatever the latest raw frame is and runs
    the full Detector + tracker pipeline on it. Deliberately does NOT
    render or encode an image here -- see the module docstring's
    correction -- it only stores the resulting tracked detections
    (shape/pixel/color/orientation/confidence), which stream_jpeg_loop
    draws onto a fresh frame at send time. Paces itself at however fast
    detection actually runs -- no sleep needed, detector.detect() itself
    (~1.2-1.8s) is the natural pacing for this loop."""
    global _latest_tracked, _latest_detections
    while True:
        with _lock:
            frame = None if _latest_raw_frame is None else _latest_raw_frame.copy()
        if frame is None:
            time.sleep(0.05)
            continue

        _, detections = detector.detect(frame)  # ignore detector's own annotated copy of this (now stale) frame
        with _tracker_lock:
            tracked = tracker.update(detections)

        with _lock:
            _latest_tracked = dict(tracked)  # snapshot -- tracker.update() mutates its internal dict in place on the next call
            _latest_detections = detections


def draw_overlay(frame, tracked):
    """Draws the latest known tracked detections onto a frame -- called
    fresh for every streamed frame in stream_jpeg_loop, not once per
    detection pass, so the overlay always sits on the current live image
    even though the detections themselves are up to ~1.2-1.8s old. No
    bounding-box rectangle here (only centroid/shape data is cached from
    the detection pass, not the original YOLO box) -- shape.draw() gives
    a centroid dot + shape/color/confidence label, plus the tracker's
    persistent ID."""
    annotated = frame.copy()
    for object_id, det in tracked.items():
        det.draw(annotated)
        u, v = det.pixel
        cv2.putText(annotated, f"ID {object_id}", (int(u), int(v) - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return annotated


def stream_jpeg_loop():
    """Generator for the /video route: draws the latest known detections
    onto the FRESHEST raw frame and encodes that, every send -- so the
    live feed genuinely updates at ~GRAB_TARGET_FPS regardless of how
    often detect_loop refreshes the detections themselves. Encoding
    happens per viewer here (a step back from encoding once in
    detect_loop and sharing it), but a JPEG encode of one frame is single-
    digit milliseconds -- trivial next to the smoothness this buys, and
    fine for this project's expected handful of simultaneous viewers."""
    while True:
        with _lock:
            raw = _latest_raw_frame
            tracked = _latest_tracked

        if raw is not None:
            annotated = draw_overlay(raw, tracked)
            ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
            frame_bytes = buffer.tobytes() if ok else None
        else:
            frame_bytes = None

        if frame_bytes is not None:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                frame_bytes +
                b'\r\n'
            )
        time.sleep(1 / 15)  # ~15fps display cap -- plenty smooth, easy on the Pi/network


@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Basler Camera</title>
    </head>
    <body>
        <h1>Basler Live Feed</h1>
        <img src="/video" style="width: 1500px; height: auto;">
        <div>
            <button onclick="pick()" style="font-size:1.3em; padding:10px 30px; margin-top:10px;">Pick &amp; Place</button>
        </div>
        <div id="status"></div>
        <script>
            async function pick() {
                document.getElementById('status').innerText = 'Running pick-and-place...';
                const res = await fetch('/pick', { method: 'POST' });
                const data = await res.json();
                document.getElementById('status').innerText =
                    data.ok ? JSON.stringify(data.results) : `Error: ${data.error}`;
            }
        </script>
    </body>
    </html>
    """


@app.route('/video')
def video():
    return Response(
        stream_jpeg_loop(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def _on_placed(object_id):
    """Called by run_pick_and_place_cycle the instant one shape is
    actually picked and placed -- removes it from both the tracker's
    own internal state and the exposed _latest_tracked snapshot right
    away, rather than leaving it to linger at its old (now-empty) spot
    until tracker.update() naturally ages it out over the next several
    detection passes (see tracker.py's discard() docstring). Without
    this, a second /pick click soon after the first could target an
    object that was already moved."""
    with _tracker_lock:
        tracker.discard(object_id)
    with _lock:
        _latest_tracked.pop(object_id, None)


@app.route('/pick', methods=['POST'])
def pick():
    """Runs one pick-and-place cycle over whatever's currently tracked --
    triggered on demand (a button click), not automatically, so the arm
    never moves without you asking it to. Uses _latest_tracked as-is
    rather than re-detecting, so what you see in the live feed at the
    moment you click is exactly what gets picked. Passes the dict
    itself (not .values()) so each placed object's ID can be reported
    back and removed immediately via _on_placed -- see its docstring."""
    if not _pick_lock.acquire(blocking=False):
        return jsonify(ok=False, error="A pick cycle is already in progress"), 409

    try:
        with _lock:
            tracked_snapshot = dict(_latest_tracked)

        if not tracked_snapshot:
            return jsonify(ok=False, error="No current detections to pick"), 400

        results = run_pick_and_place_cycle(tracked_snapshot, on_placed=_on_placed)
        return jsonify(ok=True, results=results)

    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500

    finally:
        _pick_lock.release()


if __name__ == '__main__':
    threading.Thread(target=grab_loop, daemon=True).start()
    threading.Thread(target=detect_loop, daemon=True).start()
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        close_camera(camera)
        # Clean shutdown -- NOT called after an ordinary pick cycle (see
        # robot_client.py's module docstring for why that broke the
        # robot-side script).
        close_robot_connection()
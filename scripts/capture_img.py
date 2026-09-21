"""
Web-based image capture for building the training dataset.

Rewritten to work with headless OpenCV (opencv-python-headless has no
imshow()/waitKey() -- there's no GUI backend at all, so a local preview
window isn't an option on the Pi). Instead, this runs a small Flask
server -- the same pattern as your existing pi_stream.py -- so you view
the live feed and capture frames from a browser on any device on the
network, not from the Pi's own display.

WHAT TO CAPTURE (important -- read before running)
-----------------------------------------------------
Your exam scene will likely have all 5 shapes on the table at once (the
placement board in the project brief has slots for all 5), and your
model must learn to IGNORE the 2 shapes you're not assigned, not just
never have seen them. So:

  - Include ALL 5 shapes in most frames, not just your 3 assigned ones.
    You will only label your 3 in auto_label.py -- the other 2 become
    unlabeled background, teaching the model to ignore them.
  - Vary object ORIENTATION across the full 0-360 degree range. This
    matters a lot since your pipeline needs correct orientation output.
  - Vary POSITION across your whole working area, not just the center.
  - Vary LIGHTING if you can (different times of day, lamp on/off) --
    the more lighting variety, the more robust the model.
  - Include frames with objects TOUCHING or PARTIALLY OVERLAPPING, not
    just neatly separated -- this happens in practice.
  - Capture each of your 3 assigned shapes in every color you defined
    in classes.py, if your physical objects actually come in multiple
    colors. If all your objects are one color, simplify COLORS in
    classes.py to match (having color classes the model never sees
    real examples of will hurt training, not help it).
  - Aim for AT LEAST 80-150 images total as a starting point for a
    3-class YOLO-nano model; more is better, especially more variety
    in position/orientation/lighting rather than many near-duplicate
    frames of the same setup.

USAGE
-----
Run this on the Raspberry Pi:

    python3 capture_images.py

Then open http://<pi-ip>:5000 in a browser on any device on the same
network (your laptop/phone works fine -- doesn't have to be a browser
on the Pi itself). You'll see the live feed and a Capture button;
each click saves the current frame to dataset/raw/ and logs it to
dataset/raw/manifest.csv (timestamp, filename), same as before. This
script does NOT ask you to pre-sort images by shape/color -- that
would force one shape/color per frame, which works against capturing
realistic multi-shape scenes. Auto-labeling and color detection happen
per-object in auto_label.py instead, from the actual pixels in each
image.
"""

import os
import csv
import time
import threading
from datetime import datetime

import cv2
from flask import Flask, Response, jsonify, render_template_string

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "raw")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "manifest.csv")

app = Flask(__name__)

# Shared state between the background grab loop and the Flask routes.
_lock = threading.Lock()
_latest_frame = None      # most recent raw frame (numpy array, BGR)
_latest_jpeg = None       # most recent frame, already JPEG-encoded, for streaming
_capture_count = 0


def get_frame_source():
    """Try the Basler camera (pypylon) first; fall back to a regular
    webcam so this script is also usable for a quick test off the Pi."""
    try:
        from pypylon import pylon

        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        def grab():
            result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            frame = None
            if result.GrabSucceeded():
                frame = converter.Convert(result).GetArray()
            result.Release()
            return frame

        print("Using Basler camera (pypylon)")
        return grab, camera.StopGrabbing

    except Exception as e:
        print(f"pypylon unavailable ({e}); falling back to cv2.VideoCapture(0)")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("No camera available (tried Basler and default webcam)")

        def grab():
            ok, frame = cap.read()
            return frame if ok else None

        return grab, cap.release


def grab_loop():
    """Runs in a background thread: continuously pulls frames from the
    camera and keeps the latest one (raw + JPEG-encoded) available for
    the Flask routes below, same idea as your pi_stream.py's
    generate_frames(), but decoupled from the HTTP response so /video
    and /capture can both use the same up-to-date frame."""
    global _latest_frame, _latest_jpeg
    grab, cleanup = get_frame_source()
    try:
        while True:
            frame = grab()
            if frame is None:
                continue
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            with _lock:
                _latest_frame = frame
                _latest_jpeg = buffer.tobytes()
    finally:
        cleanup()


def mjpeg_stream():
    while True:
        with _lock:
            jpeg = _latest_jpeg
        if jpeg is not None:
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(0.05)  # ~20fps cap; plenty for a capture UI, easy on the Pi


PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Dataset Capture</title>
    <style>
        body { font-family: sans-serif; text-align: center; background: #111; color: #eee; }
        img { max-width: 95vw; border: 2px solid #444; margin-top: 10px; }
        button {
            font-size: 1.5em; padding: 15px 40px; margin-top: 20px;
            background: #2a7; color: white; border: none; border-radius: 8px; cursor: pointer;
        }
        button:active { background: #185; }
        #status { margin-top: 10px; font-size: 1.1em; }
    </style>
</head>
<body>
    <h2>Dataset Capture &mdash; {{ count }} images so far</h2>
    <img src="/video" />
    <div>
        <button onclick="capture()">Capture (or press SPACE)</button>
    </div>
    <div id="status"></div>
    <script>
        async function capture() {
            const res = await fetch('/capture', { method: 'POST' });
            const data = await res.json();
            document.getElementById('status').innerText =
                data.ok ? `Saved ${data.filename}  (total: ${data.count})` : `Error: ${data.error}`;
        }
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space') { e.preventDefault(); capture(); }
        });
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE, count=_capture_count)


@app.route("/video")
def video():
    return Response(mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/capture", methods=["POST"])
def capture():
    global _capture_count
    with _lock:
        frame = None if _latest_frame is None else _latest_frame.copy()

    if frame is None:
        return jsonify(ok=False, error="No frame available yet"), 503

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"img_{timestamp}.jpg"
    filepath = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(filepath, frame)

    with open(MANIFEST_PATH, "a", newline="") as manifest:
        csv.writer(manifest).writerow([timestamp, filename])

    _capture_count += 1
    print(f"Saved {filename}  (total: {_capture_count})")
    return jsonify(ok=True, filename=filename, count=_capture_count)


def main():
    global _capture_count

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "w", newline="") as manifest:
            csv.writer(manifest).writerow(["timestamp", "filename"])

    _capture_count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".jpg")])
    print(f"Starting with {_capture_count} existing images in {OUTPUT_DIR}")

    threading.Thread(target=grab_loop, daemon=True).start()

    print("Open http://<this-device-ip>:5000 in a browser to start capturing")
    app.run(host="0.0.0.0", port=5000, threaded=True)


if __name__ == "__main__":
    main()
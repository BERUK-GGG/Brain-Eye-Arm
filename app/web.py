import time

import cv2
from flask import Flask, Response, jsonify

from app.overlay import draw_overlay
from app.service import VisionService
from robot.picker import PickAndPlaceController, PickInProgress

STREAM_JPEG_QUALITY = 70    # 0-100; lower = smaller frames, more visible artifacts
STREAM_FPS = 15

INDEX_HTML = """
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


def create_app(service: VisionService, controller: PickAndPlaceController) -> Flask:
    app = Flask(__name__)

    def mjpeg_frames():
        """Draws the latest detections onto the FRESHEST frame for every
        send, so the picture stays live even though detections lag."""
        while True:
            frame = service.latest_frame()
            if frame is not None:
                annotated = draw_overlay(frame, service.tracked_snapshot())
                ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            time.sleep(1 / STREAM_FPS)

    @app.route("/")
    def index():
        return INDEX_HTML

    @app.route("/video")
    def video():
        return Response(mjpeg_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/pick", methods=["POST"])
    def pick():
        """One pick-and-place cycle over whatever is tracked right now.
        Runs on demand only, so the arm never moves unasked."""
        tracked = service.tracked_snapshot()
        if not tracked:
            return jsonify(ok=False, error="No current detections to pick"), 400

        try:
            results = controller.run_cycle(tracked, on_placed=service.discard)
        except PickInProgress as e:
            return jsonify(ok=False, error=str(e)), 409
        except Exception as e:
            return jsonify(ok=False, error=str(e)), 500
        return jsonify(ok=True, results=results)

    return app

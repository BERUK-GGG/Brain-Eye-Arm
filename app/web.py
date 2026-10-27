import time
from typing import List

import cv2
from flask import Flask, Response, jsonify, request

from app.overlay import draw_overlay
from app.selection import filter_tracked
from app.service import VisionService
from robot.picker import PickAndPlaceController, PickInProgress

STREAM_JPEG_QUALITY = 70    # 0-100; lower = smaller frames, more visible artifacts
STREAM_FPS = 15

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Basler Camera</title>
    <style>
        button { font-size: 1.1em; padding: 10px 24px; margin: 6px 6px 0 0; }
        #status { margin-top: 12px; font-family: monospace; }
    </style>
</head>
<body>
    <h1>Basler Live Feed</h1>
    <img src="/video" style="width: 1500px; height: auto;">
    <div>
        <p>
            Color:
            <select id="color-select"><option value="">Any color</option></select>
        </p>
        <p>Pick one object -- which shape gets picked next is chosen here, not automatically:</p>
        <div id="shape-buttons"></div>
        <p>Or, to pick everything currently tracked (matching the color above) in one cycle:</p>
        <button onclick="pick(null)">Pick All</button>
    </div>
    <div id="status"></div>
    <script>
        async function loadOptions(url, elementId, label) {
            const res = await fetch(url);
            const data = await res.json();
            const container = document.getElementById(elementId);
            for (const value of data[label]) {
                if (elementId === 'color-select') {
                    const option = document.createElement('option');
                    option.value = value;
                    option.textContent = value;
                    container.appendChild(option);
                } else {
                    const button = document.createElement('button');
                    button.textContent = 'Pick ' + value;
                    button.onclick = () => pick(value);
                    container.appendChild(button);
                }
            }
        }

        async function pick(shape) {
            const color = document.getElementById('color-select').value || null;
            const label = [shape, color].filter(Boolean).join(' ') || 'everything tracked';
            document.getElementById('status').innerText = `Picking ${label}...`;
            const res = await fetch('/pick', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ shape: shape, color: color }),
            });
            const data = await res.json();
            document.getElementById('status').innerText =
                data.ok ? JSON.stringify(data.results) : `Error: ${data.error}`;
        }

        loadOptions('/shapes', 'shape-buttons', 'shapes');
        loadOptions('/colors', 'color-select', 'colors');
    </script>
</body>
</html>
"""


def create_app(service: VisionService, controller: PickAndPlaceController, assigned_shapes: List[str],
                known_colors: List[str] = ()) -> Flask:
    """assigned_shapes: the shape names this GUI offers buttons for (see
    config.json's place_targets). known_colors: the color names its
    dropdown offers (vision/color.py's PALETTE). Both keep the page in
    sync with whatever this config is actually set up for, rather than
    hardcoding names into the page."""
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

    @app.route("/shapes")
    def shapes():
        return jsonify(shapes=assigned_shapes)

    @app.route("/colors")
    def colors():
        return jsonify(colors=known_colors)

    @app.route("/pick", methods=["POST"])
    def pick():
        """One pick-and-place cycle over the current selection. Runs on
        demand only, so the arm never moves unasked.

        Body (JSON, all optional): {"shape": "Triangle", "color": "red"}.
        Either omitted or null matches anything; an empty body (or none
        at all) picks everything currently tracked, same as the old
        "Pick All" -- so this stays backward compatible with a plain
        POST /pick."""
        payload = request.get_json(silent=True) or {}
        shape = payload.get("shape")
        color = payload.get("color")

        tracked = filter_tracked(service.tracked_snapshot(), shape=shape, color=color)
        if not tracked:
            wanted = " ".join(filter(None, [color, shape])) or "anything"
            return jsonify(ok=False, error=f"No current detection matching '{wanted}' to pick"), 400

        try:
            results = controller.run_cycle(tracked, on_placed=service.discard)
        except PickInProgress as e:
            return jsonify(ok=False, error=str(e)), 409
        except Exception as e:
            return jsonify(ok=False, error=str(e)), 500
        return jsonify(ok=True, results=results)

    return app

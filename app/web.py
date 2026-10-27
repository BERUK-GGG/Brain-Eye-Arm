import time
from typing import List, Optional

import cv2
from flask import Flask, Response, jsonify, render_template, request

from app.overlay import draw_overlay
from app.pick_worker import PickWorker, WorkerBusy
from app.selection import filter_tracked
from app.service import VisionService
from robot.client import RobotClient

STREAM_JPEG_QUALITY = 70    # 0-100; lower = smaller frames, more visible artifacts
STREAM_FPS = 10
STREAM_WIDTH = 1280         # px; the page shows it ~1000 px wide, so full 2448 px only cost CPU


def create_app(service: VisionService, worker: PickWorker, assigned_shapes: List[str],
                known_colors: List[str] = (), robot: Optional[RobotClient] = None) -> Flask:
    """The HMI's HTTP layer only: it reads the vision service and hands pick
    jobs to the worker's robot thread (app/pick_worker.py), never driving
    the robot itself.

    assigned_shapes: the shape names this GUI offers (see config.json's
    place_targets). known_colors: the color names it offers (see
    vision/color.py's PALETTE). Both keep the page in sync with whatever
    this config is actually set up for, rather than hardcoding names into
    the page. robot is optional: without it the status bar shows no link
    state."""
    app = Flask(__name__)

    def mjpeg_frames():
        """Draws the latest detections onto the FRESHEST frame for every
        send, so the picture stays live even though detections lag.

        The frame is downscaled to STREAM_WIDTH before drawing and JPEG
        encoding (encoding a full 5 MP frame is the single most expensive
        thing the Pi did per frame, and it competed with inference), and a
        frame is only re-encoded when the picture or detections changed."""
        last_frame, last_tracked = None, None
        while True:
            frame = service.latest_frame()
            tracked = service.tracked_snapshot()
            if frame is not None and (frame is not last_frame or tracked != last_tracked):
                last_frame, last_tracked = frame, tracked
                scale = min(1.0, STREAM_WIDTH / frame.shape[1])
                # LINEAR, not AREA: ~6x cheaper, and indistinguishable at this ~0.5 scale.
                small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) if scale < 1 else frame
                annotated = draw_overlay(small, tracked, scale=scale)
                ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            time.sleep(1 / STREAM_FPS)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video")
    def video():
        return Response(mjpeg_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/shapes")
    def shapes():
        return jsonify(shapes=assigned_shapes)

    @app.route("/colors")
    def colors():
        return jsonify(colors=list(known_colors))

    @app.route("/detections")
    def detections():
        """Currently tracked parts, by tracker ID."""
        return jsonify(parts=[
            {
                "id": object_id,
                "shape": shape.shape_name,
                "color": shape.color,
                "confidence": shape.confidence,
                "orientation": shape.orientation if shape.orientation_points is not None else None,
                "pixel": shape.pixel,
                "robot": shape.robot,
            }
            for object_id, shape in sorted(service.tracked_snapshot().items())
        ])

    @app.route("/status")
    def status():
        return jsonify(
            vision=service.stats(),
            robot={
                "address": robot.address if robot else None,
                "connected": robot.connected if robot else None,
                "busy": worker.busy,
                "current": worker.current_request,
            },
            last_pick=worker.last_outcome(),
        )

    @app.route("/history")
    def pick_history():
        """Handled objects since startup, oldest first; ?after=N returns
        only entries numbered after N."""
        after = request.args.get("after", default=0, type=int)
        return jsonify(entries=worker.history_since(after))

    @app.route("/pick", methods=["POST"])
    def pick():
        """Queues one pick-and-place cycle over the current selection and
        returns at once (202 with a job ID); the robot thread runs it, and
        /status reports the outcome as last_pick with that job ID. Runs on
        demand only, so the arm never moves unasked.

        Body (JSON, all optional): {"shape": "Triangle", "color": "red"},
        or {"object_id": 3} for one specific tracked part. Omitted or null
        fields match anything; an empty body (or none at all) picks
        everything currently tracked -- so a plain POST /pick still works."""
        payload = request.get_json(silent=True) or {}
        shape = payload.get("shape")
        color = payload.get("color")
        object_id = payload.get("object_id")

        snapshot = service.tracked_snapshot()
        if object_id is not None:
            wanted = f"part #{object_id}"
            tracked = {object_id: snapshot[object_id]} if object_id in snapshot else {}
        else:
            wanted = " ".join(filter(None, [color, shape])) or "everything tracked"
            tracked = filter_tracked(snapshot, shape=shape, color=color)

        if not tracked:
            print(f"Pick {wanted}: nothing currently detected matches")
            return jsonify(ok=False, error=f"No current detection matching '{wanted}' to pick"), 400
        try:
            job_id = worker.submit(wanted, tracked)
        except WorkerBusy as e:
            print(f"Pick {wanted}: refused -- {e}")
            return jsonify(ok=False, error=str(e)), 409
        return jsonify(ok=True, job=job_id, request=wanted), 202

    return app

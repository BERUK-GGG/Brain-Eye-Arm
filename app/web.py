import threading
import time
from typing import List, Optional

import cv2
from flask import Flask, Response, jsonify, render_template, request

from app.logbuffer import LogBuffer
from app.overlay import draw_overlay
from app.selection import filter_tracked
from app.service import VisionService
from robot.client import RobotClient
from robot.picker import PickAndPlaceController, PickInProgress

STREAM_JPEG_QUALITY = 70    # 0-100; lower = smaller frames, more visible artifacts
STREAM_FPS = 15


def create_app(service: VisionService, controller: PickAndPlaceController, assigned_shapes: List[str],
                known_colors: List[str] = (), robot: Optional[RobotClient] = None,
                logs: Optional[LogBuffer] = None) -> Flask:
    """assigned_shapes: the shape names this GUI offers (see config.json's
    place_targets). known_colors: the color names it offers (see
    vision/color.py's PALETTE). Both keep the page in sync with whatever
    this config is actually set up for, rather than hardcoding names into
    the page. robot and logs are optional: without them the status bar
    shows no link state and the log panel stays empty."""
    app = Flask(__name__)

    last_pick = {}              # outcome of the most recent /pick, for /status
    last_pick_lock = threading.Lock()

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
        with last_pick_lock:
            pick = dict(last_pick)
        return jsonify(
            vision=service.stats(),
            robot={
                "address": robot.address if robot else None,
                "connected": robot.connected if robot else None,
                "busy": controller.busy,
            },
            last_pick=pick or None,
        )

    @app.route("/logs")
    def log_lines():
        """Printed lines numbered after ?after=N (0 = everything kept)."""
        after = request.args.get("after", default=0, type=int)
        return jsonify(lines=logs.since(after) if logs else [])

    @app.route("/pick", methods=["POST"])
    def pick():
        """One pick-and-place cycle over the current selection. Runs on
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

        def finish(code: int, **outcome):
            with last_pick_lock:
                last_pick.clear()
                last_pick.update(time=time.strftime("%H:%M:%S"), request=wanted, **outcome)
            return jsonify(**outcome), code

        if not tracked:
            print(f"Pick {wanted}: nothing currently detected matches")
            return finish(400, ok=False, error=f"No current detection matching '{wanted}' to pick")

        print(f"Pick {wanted}: {len(tracked)} part(s)")
        try:
            results = controller.run_cycle(tracked, on_placed=service.discard)
        except PickInProgress as e:
            print(f"Pick {wanted}: refused -- {e}")
            return jsonify(ok=False, error=str(e)), 409  # not recorded: the running cycle's outcome matters
        except Exception as e:
            print(f"ERROR: pick {wanted} failed: {e}")
            return finish(500, ok=False, error=str(e))
        print(f"Pick {wanted}: done -- " + ", ".join(f"{r['shape']} {r['status']}" for r in results))
        return finish(200, ok=True, results=results)

    return app

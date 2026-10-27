import itertools
import threading
import time
from collections import deque
from typing import List, Optional

import cv2
from flask import Flask, Response, jsonify, render_template, request

from app.overlay import draw_overlay
from app.selection import filter_tracked, find_same_object
from app.service import VisionService
from robot.client import RobotClient
from robot.picker import PickAndPlaceController, PickInProgress

STREAM_JPEG_QUALITY = 70    # 0-100; lower = smaller frames, more visible artifacts
STREAM_FPS = 10
STREAM_WIDTH = 1280         # px; the page shows it ~1000 px wide, so full 2448 px only cost CPU
HISTORY_LENGTH = 500        # picked objects kept for the HMI's history table (this run only)


def create_app(service: VisionService, controller: PickAndPlaceController, assigned_shapes: List[str],
                known_colors: List[str] = (), robot: Optional[RobotClient] = None) -> Flask:
    """assigned_shapes: the shape names this GUI offers (see config.json's
    place_targets). known_colors: the color names it offers (see
    vision/color.py's PALETTE). Both keep the page in sync with whatever
    this config is actually set up for, rather than hardcoding names into
    the page. robot is optional: without it the status bar shows no link
    state."""
    app = Flask(__name__)

    last_pick = {}              # outcome of the most recent /pick, for /status
    history = deque(maxlen=HISTORY_LENGTH)  # one entry per handled object, plus failed requests
    entry_numbers = itertools.count(1)
    state_lock = threading.Lock()           # guards last_pick and history

    def record(entry: dict) -> None:
        with state_lock:
            history.append({"seq": next(entry_numbers), "time": time.strftime("%H:%M:%S"), **entry})

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
        with state_lock:
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

    @app.route("/history")
    def pick_history():
        """Handled objects since startup, oldest first; ?after=N returns
        only entries numbered after N."""
        after = request.args.get("after", default=0, type=int)
        with state_lock:
            return jsonify(entries=[e for e in history if e["seq"] > after])

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
            with state_lock:
                last_pick.clear()
                last_pick.update(time=time.strftime("%H:%M:%S"), request=wanted, **outcome)
            return jsonify(**outcome), code

        if not tracked:
            print(f"Pick {wanted}: nothing currently detected matches")
            return finish(400, ok=False, error=f"No current detection matching '{wanted}' to pick")

        def find_again(shape):
            """Is the part still lying where it was? (arm is home, out of view)"""
            return find_same_object(shape, service.redetect())

        if controller.busy:
            print(f"Pick {wanted}: refused -- a pick cycle is already in progress")
            return jsonify(ok=False, error="A pick cycle is already in progress"), 409

        print(f"Pick {wanted}: {len(tracked)} part(s)")
        # No tracker updates while the arm is in view -- see VisionService.
        service.pause()
        try:
            results = controller.run_cycle(tracked, on_placed=service.discard, on_result=record,
                                           find_again=find_again)
        except PickInProgress as e:
            print(f"Pick {wanted}: refused -- {e}")
            return jsonify(ok=False, error=str(e)), 409  # not recorded: the running cycle's outcome matters
        except Exception as e:
            print(f"ERROR: pick {wanted} failed: {e}")
            record({"status": "failed", "request": wanted, "error": str(e)})
            return finish(500, ok=False, error=str(e))
        finally:
            if not controller.busy:  # a refused request mustn't resume during someone else's cycle
                service.resume()
        print(f"Pick {wanted}: done -- " + ", ".join(f"{r['shape']} {r['status']}" for r in results))
        return finish(200, ok=True, results=results)

    return app

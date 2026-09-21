"""
Robot TCP client: connects to the Dobot TCP server (dobot_tcp_server.py,
running on the robot controller) and drives a pick-and-place cycle for a
list of already-detected shapes.

WHY THIS IS ITS OWN MODULE, SEPARATE FROM STREAM.PY AND THE OLD
STANDALONE dobot_tcp_client.py
--------------------------------------------------------------------
The earlier dobot_tcp_client.py ran as its own script: its own camera
connection, its own loaded YOLO model, its own single detect-then-pick
pass. Running it as a second, separate process meant fighting stream.py
over the one camera connection Basler/pylon allows, loading the model
twice, and picking from a fresh one-shot detection instead of whatever
stream.py's already-running detect_loop knows. This module holds ONLY
the robot-communication side (TCP connect, POSE/DO messages, the
pick-and-place sequence) -- it takes a list of DetectedShape instances
and doesn't care where they came from. stream.py's /pick route supplies
them from its live _latest_tracked state; the (now much thinner)
standalone dobot_tcp_client.py still exists for testing the robot link
on its own, without the rest of stream.py running, and supplies them
from its own one-shot camera grab.
"""

import socket
import time

HOST = "192.168.5.1"
PORT = 6601
TIMEOUT = 5

MESSAGE = "Connection Established"

XY_HOME = ["200", "-300"]

Z_safe = "50"
Z_pick = "40"

# Suction (vacuum gripper) digital output. DO index 1, ON=1/OFF=0 is the
# typical Dobot default -- update these three if your actual wiring uses
# a different DO index or polarity.
DO_SUCTION_INDEX = 1
DO_ON = 1
DO_OFF = 0

# How long to hold at Z_pick after switching suction on/off before
# raising back to Z_safe -- gives the vacuum a moment to actually grip or
# release before the arm moves away. Tune to your actual gripper/object
# weight if picks are unreliable.
SUCTION_SETTLE_SECONDS = 0.5


def _recv_line(sock):
    data = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Server closed the connection")

        data += chunk
        if b"\n" in data:
            line, _, rest = data.partition(b"\n")
            sock.setblocking(False)
            try:
                extra = sock.recv(4096)
                data = (extra + rest) if extra else rest
            except socket.error:
                data = rest
            finally:
                sock.settimeout(TIMEOUT)
            return line.decode("utf-8", errors="replace").strip()


def _send_pose(sock, x, y, z):
    message = f"POSE,{x},{y},{z}\n"
    print(f"Sending: {message.strip()}")
    sock.sendall(message.encode("utf-8"))
    response = _recv_line(sock)
    print(f"Server reply: {response}")
    return response


def _send_do(sock, do_index, state):
    message = f"DO,{do_index},{state}\n"
    print(f"Sending: {message.strip()}")
    sock.sendall(message.encode("utf-8"))
    response = _recv_line(sock)
    print(f"Server reply: {response}")
    return response


def _pick_and_place_one(sock, shape):
    """Full cycle for one detected shape: pick it up from where it was
    detected, place it at its configured target, return home. Position
    only -- orientation is measured and reported but not sent to the
    robot; doing that would mean extending the "POSE" message (e.g.
    "POSE,x,y,z,r") on BOTH this client AND dobot_tcp_server.py itself, a
    deliberate follow-up, not done here."""
    x, y = shape.robot
    place_x, place_y = shape.get_place_target()  # raises if this shape has no target -- caller should check first

    # -- Pick --
    _send_pose(sock, x, y, Z_safe)
    _send_pose(sock, x, y, Z_pick)
    _send_do(sock, DO_SUCTION_INDEX, DO_ON)
    time.sleep(SUCTION_SETTLE_SECONDS)
    _send_pose(sock, x, y, Z_safe)

    # -- Place --
    _send_pose(sock, place_x, place_y, Z_safe)
    _send_pose(sock, place_x, place_y, Z_pick)
    _send_do(sock, DO_SUCTION_INDEX, DO_OFF)
    time.sleep(SUCTION_SETTLE_SECONDS)
    _send_pose(sock, place_x, place_y, Z_safe)

    # -- Return home --
    _send_pose(sock, XY_HOME[0], XY_HOME[1], Z_safe)


def run_pick_and_place_cycle(detections):
    """Connects to the robot, picks and places each detection in turn
    (sorted for a repeatable order), then disconnects. A shape with no
    configured place target (see config.json's "place_targets") is
    skipped entirely rather than picked up with nowhere to go.

    detections: list of DetectedShape (e.g. stream.py's
    _latest_tracked.values(), or a fresh single-frame detection).

    Returns a list of per-shape result dicts, e.g.:
        [{"shape": "Triangle", "color": "red", "status": "placed"},
         {"shape": "Square", "color": "blue", "status": "skipped_no_target"}]
    Raises socket.timeout / socket.error / ConnectionError on a
    communication failure -- callers decide how to report that.
    """
    results = []
    if not detections:
        return results

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(TIMEOUT)
    try:
        print(f"Connecting to {HOST}:{PORT}...")
        sock.connect((HOST, PORT))
        print("Connected")

        sock.sendall((MESSAGE + "\n").encode("utf-8"))
        print(f"Server says: {_recv_line(sock)}")

        for shape in sorted(detections, key=lambda s: s.pixel):
            print(
                f"Detected {shape.shape_name} ({shape.color}) at {shape.robot} "
                f"with orientation {shape.orientation:.2f} degrees (not applied to robot -- see module docstring)"
            )

            try:
                shape.get_place_target()
            except NotImplementedError:
                print(f"Skipping {shape.shape_name}: no place target configured in config.json -- "
                      f"nowhere defined to put it down.")
                results.append({"shape": shape.shape_name, "color": shape.color, "status": "skipped_no_target"})
                continue

            _pick_and_place_one(sock, shape)
            results.append({"shape": shape.shape_name, "color": shape.color, "status": "placed"})

    finally:
        sock.close()

    return results
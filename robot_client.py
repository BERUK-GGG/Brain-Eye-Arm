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

WHY THE CONNECTION IS PERSISTENT (an actual bug fix, not just style)
--------------------------------------------------------------------
dobot_tcp_server.py -- the script running ON the robot controller --
loops reading commands until a TCPRead error, which happens when the
client disconnects, and THEN calls TCPDestroy() and ends:

    while True:
        err, recBuf = TCPRead(socket)
        if err != 0:
            break
        ...
    TCPDestroy(socket)

An earlier version of this module opened a fresh socket, ran one pick
cycle, and closed it again -- once per /pick click. Every single click
was making the robot think the client had disconnected, which broke its
loop and ended the WHOLE Application script on the robot, requiring a
manual restart via DobotStudio Pro before the next pick could work at
all. The fix: hold ONE connection open for as long as stream.py runs,
and reuse it across every pick cycle -- see _get_connection() /
close_connection() below. Only close it on a real communication failure
(so the next call reconnects cleanly) or on clean shutdown.
"""

import socket
import threading
import time

HOST = "192.168.5.1"
PORT = 6601
TIMEOUT = 5

MESSAGE = "Connection Established"

XY_HOME = ["250", "-170"]

Z_safe = "100"
Z_pick = "17"

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

# The one persistent connection to the robot, shared across every
# run_pick_and_place_cycle() call -- see the module docstring for why
# this must NOT be opened and closed per call. Guarded by _lock so
# concurrent callers (e.g. stream.py's /pick plus the standalone
# dobot_tcp_client.py, if both somehow ran at once) can't interleave
# commands on the same socket.
_lock = threading.Lock()
_sock = None


def _connect():
    """Opens a fresh connection and does the initial handshake. Only
    called when there's no live connection to reuse."""
    global _sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(TIMEOUT)
    print(f"Connecting to {HOST}:{PORT}...")
    sock.connect((HOST, PORT))
    print("Connected")
 
    sock.sendall((MESSAGE + "\n").encode("utf-8"))
    print(f"Server says: {_recv_line(sock)}")
 
    _sock = sock
    return sock
 
 
def _get_connection():
    """Returns the persistent connection, connecting (or reconnecting,
    if a previous failure dropped it) as needed."""
    if _sock is None:
        return _connect()
    return _sock
 
 
def close_connection(notify=True):
    """Closes the persistent connection, if one is open. Call this on
    CLEAN SHUTDOWN (e.g. stream.py's `finally` block) or after a
    communication failure so the next call reconnects fresh -- never
    after an ordinary successful pick cycle (see module docstring).
 
    notify: if True (the default -- use this for a clean, intentional
    shutdown), sends an explicit "CLOSE" message first, so
    dobot_tcp_server.py actually breaks its own loop and ends the
    Application script -- it no longer treats an ordinary dropped
    connection as a signal to stop (see that script's own comments),
    so without this it would just sit there waiting forever, alive but
    orphaned, until manually stopped in DobotStudio Pro. Pass False
    when the connection is already known-broken (e.g. after a
    send/recv failure) -- there's nothing to notify at that point, and
    trying would just raise another exception here."""
    global _sock
    if _sock is not None:
        if notify:
            try:
                _sock.sendall(b"CLOSE\n")
            except OSError:
                pass  # best-effort -- we're closing our side regardless
        try:
            _sock.close()
        except OSError:
            pass
        _sock = None
 
 
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
 
 
def run_pick_and_place_cycle(detections, on_placed=None):
    """Picks and places each detection in turn (sorted for a repeatable
    order), reusing the ONE persistent connection to the robot (see the
    module docstring for why -- closing it after every cycle breaks the
    robot-side script). A shape with no configured place target (see
    config.json's "place_targets") is skipped entirely rather than
    picked up with nowhere to go.
 
    detections: EITHER a dict of object_id -> DetectedShape (e.g.
    stream.py's _latest_tracked -- pass the dict itself, not
    .values(), so this can report back WHICH object_id got placed),
    OR a plain list of DetectedShape (e.g. a fresh one-shot detection
    with no tracker/IDs involved, as in the standalone
    dobot_tcp_client.py) -- for a list, object_id is just None
    throughout, and on_placed is never called.
 
    on_placed(object_id): optional callback invoked immediately after
    a shape is successfully picked AND placed -- BEFORE moving on to
    the next one, not batched at the end. stream.py's /pick route uses
    this to remove that object from its own tracked state right away.
    Without this, a picked-and-moved object keeps sitting in whatever
    tracking state the caller keeps, still looking "there" at its old,
    now-empty spot, until it naturally ages out from no longer being
    redetected -- which can be many seconds, long enough for a second
    /pick click to try to pick the same (now already-moved, or empty)
    spot again. Only called when detections is a dict (a list has no
    IDs to report).
 
    Returns a list of per-shape result dicts, e.g.:
        [{"object_id": 3, "shape": "Triangle", "color": "red", "status": "placed"},
         {"object_id": 5, "shape": "Square", "color": "blue", "status": "skipped_no_target"}]
    ("object_id" is None throughout when detections was a plain list.)
    Raises socket.timeout / socket.error / ConnectionError on a
    communication failure -- callers decide how to report that. On any
    such failure, the connection is dropped so the NEXT call reconnects
    cleanly rather than reusing a socket that's now in an unknown state.
    """
    results = []
    if not detections:
        return results
 
    if isinstance(detections, dict):
        items = list(detections.items())
    else:
        items = [(None, shape) for shape in detections]
 
    if not items:
        return results
 
    with _lock:
        try:
            sock = _get_connection()
 
            for object_id, shape in sorted(items, key=lambda item: item[1].pixel):
                print(
                    f"Detected {shape.shape_name} ({shape.color}) at {shape.robot} "
                    f"with orientation {shape.orientation:.2f} degrees (not applied to robot -- see module docstring)"
                )
 
                try:
                    shape.get_place_target()
                except NotImplementedError:
                    print(f"Skipping {shape.shape_name}: no place target configured in config.json -- "
                          f"nowhere defined to put it down.")
                    results.append({"object_id": object_id, "shape": shape.shape_name,
                                     "color": shape.color, "status": "skipped_no_target"})
                    continue
 
                _pick_and_place_one(sock, shape)
                results.append({"object_id": object_id, "shape": shape.shape_name,
                                 "color": shape.color, "status": "placed"})
 
                if on_placed is not None and object_id is not None:
                    on_placed(object_id)
 
        except (OSError, ConnectionError):
            # Connection is now in an unknown/broken state -- drop it so
            # the next call opens a fresh one instead of reusing
            # something half-dead. notify=False: don't try to send a
            # CLOSE message over a connection that just failed -- the
            # robot-side script no longer treats a plain drop as fatal
            # anyway (see dobot_tcp_server.py), so it'll just sit idle
            # until the next client connects.
            close_connection(notify=False)
            raise
 
    return results
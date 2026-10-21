"""
TCP client for server_script/dobot_tcp_server.py (runs on the Dobot).

Protocol: newline-terminated text; every message gets exactly one reply
line.
    Connection Established -> READY
    POSE,x,y,z,r            -> ACK:POSE
    DO,index,0|1           -> ACK:DO
    CLOSE                  -> ACK:CLOSE   (ends the robot script)
    anything invalid       -> ERR:...

r is a rotation OFFSET in degrees, not an absolute angle -- the server
adds it to however P1 was originally taught (see that script's own
comments for why: this client never reads back P1's actual taught
rotation, so it can only ever say "turn r degrees from your own
reference," never "go to this absolute world angle"). r=0 always means
"whatever DobotStudio Pro's P1 point was taught with."

The connection is persistent. The robot script keeps running across
clients, so a pick cycle must not open and close a socket each time; the
connection is dropped only after a communication failure (so the next
call reconnects cleanly) or on an explicit close().
"""

import socket
import threading
from contextlib import contextmanager
from typing import Optional

from config import RobotSettings

HANDSHAKE = "Connection Established"


class RobotCommandError(RuntimeError):
    """The robot answered a command with ERR:... (the arm did not act on it)."""


class RobotClient:
    def __init__(self, settings: RobotSettings):
        self._settings = settings
        self._sock: Optional[socket.socket] = None
        self._rx = b""                      # bytes received but not yet consumed as a line
        self._lock = threading.RLock()      # one command sequence on the socket at a time

    @contextmanager
    def session(self):
        """Holds the connection exclusively for a multi-command sequence,
        connecting if needed. On a communication failure the connection is
        dropped and the error re-raised, so the next session starts fresh."""
        with self._lock:
            try:
                self._ensure_connected()
                yield self
            except OSError:
                self.close(notify=False)  # nothing to notify: the socket is already broken
                raise

    def move_to(self, x, y, z, r=0) -> str:
        """r: rotation OFFSET in degrees from P1's taught orientation (see
        module docstring) -- 0 means "leave it exactly as taught"."""
        return self._request(f"POSE,{x},{y},{z},{r}")

    def set_output(self, index: int, on: bool) -> str:
        return self._request(f"DO,{index},{1 if on else 0}")

    def close(self, notify: bool = True) -> None:
        """Drops the connection. notify=True (clean shutdown) first sends
        CLOSE so the robot script actually ends; without it the script
        would wait forever for a new client. Pass False when the socket is
        already known-broken."""
        with self._lock:
            if self._sock is None:
                return
            if notify:
                try:
                    self._sock.sendall(b"CLOSE\n")
                except OSError:
                    pass  # best effort; we're closing our side regardless
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
            self._rx = b""

    def _ensure_connected(self) -> None:
        if self._sock is not None:
            # A kept-open socket may have gone stale between cycles (robot
            # script restarted or re-listening). Probe it with the
            # handshake -- harmless, the arm doesn't move -- and reconnect
            # if it's dead, rather than failing mid-sequence.
            try:
                self._request(HANDSHAKE)
                return
            except OSError as e:
                print(f"Stale robot connection ({e}); reconnecting...")
                self.close(notify=False)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self._settings.timeout)
        print(f"Connecting to {self._settings.host}:{self._settings.port}...")
        try:
            sock.connect((self._settings.host, self._settings.port))
            self._sock, self._rx = sock, b""
            print(f"Connected. Server says: {self._request(HANDSHAKE)}")
        except OSError:
            self._sock = None
            sock.close()
            raise

    def _request(self, message: str) -> str:
        with self._lock:
            if self._sock is None:
                raise ConnectionError("Not connected to the robot")
            print(f"Sending: {message}")
            self._sock.sendall((message + "\n").encode("utf-8"))
            reply = self._read_line()
            print(f"Server reply: {reply}")
            if reply.startswith("ERR"):
                raise RobotCommandError(f"Robot rejected '{message}': {reply}")
            return reply

    def _read_line(self) -> str:
        while b"\n" not in self._rx:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("Server closed the connection")
            self._rx += chunk
        line, _, self._rx = self._rx.partition(b"\n")
        return line.decode("utf-8", errors="replace").strip()

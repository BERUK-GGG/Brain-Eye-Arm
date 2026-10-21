"""
Step-by-step check of the TCP link to server_script/dobot_tcp_server.py,
to see exactly where a "connection reset" happens. Moves nothing unless
--move is given (then one POSE to home at z_safe).

Stop main.py first, so this is the only client.

    python -m stand_alone_scripts.robot_probe [--move]
"""

import socket
import sys

from config import Config


def main():
    s = Config.load().robot
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(s.timeout)

    def step(name, fn):
        print(f"[{name}] ...", end=" ", flush=True)
        try:
            out = fn()
            print(f"OK {out!r}" if out is not None else "OK")
        except OSError as e:
            print(f"FAILED: {type(e).__name__}: {e}")
            sys.exit(1)

    def request(msg):
        sock.sendall((msg + "\n").encode())
        return sock.recv(4096).decode(errors="replace").strip()

    step(f"connect {s.host}:{s.port}", lambda: sock.connect((s.host, s.port)))
    step("handshake", lambda: request("Connection Established"))
    step("DO off (suction)", lambda: request(f"DO,{s.suction_do_index},0"))
    if "--move" in sys.argv:
        x, y = s.home_xy
        step("POSE home", lambda: request(f"POSE,{x},{y},{s.z_safe},0"))
    sock.close()  # no CLOSE sent: the robot script keeps running
    print("All steps passed.")


if __name__ == "__main__":
    main()

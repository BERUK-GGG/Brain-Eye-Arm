# version: Python3
#
# Runs ON the robot controller, inside DobotStudio Pro's Application tab.
#
# CHANGE FROM THE ORIGINAL VERSION OF THIS SCRIPT
# -------------------------------------------------
# Previously: `while True: err, recBuf = TCPRead(socket); if err != 0: break`
# -- meaning ANY TCPRead error ended the loop, called TCPDestroy(), and
# stopped the whole script. In practice this fired on every ordinary
# client disconnect, so every single pick-and-place cycle from the Pi
# side (which used to open a fresh socket, do one cycle, and close it)
# killed this Application script and required a manual restart from
# DobotStudio Pro before the next pick could work at all.
#
# Now: a TCPRead error no longer ends the script by itself. The loop
# just logs it and keeps waiting. The ONLY thing that now closes the
# connection / destroys the socket / ends the script is an explicit
# "CLOSE" (or "close_connection") message from the client -- see
# CLOSE_COMMANDS below. robot_client.py's close_connection() sends this
# on a clean, intentional shutdown; it does NOT send it after an
# ordinary pick cycle, and does NOT send it when dropping a connection
# that already failed (nothing to send to -- the socket's already dead
# on that side).
#
# RECONNECTS
# ----------
# TCPStart accepts only one client per socket. Looping back on the same
# handle after a disconnect left it dead, so every later client got
# "connection reset by peer". On a read error (or EOF) the socket is now
# destroyed and re-created, so a new client can always connect.
#
# ROTATION -- CONFIRMED ON HARDWARE
# ------------------------------------------------------
# POSE now carries a 4th value, r -- see robot/client.py's docstring for
# why it's an OFFSET from however P1 was taught, not an absolute angle.
# Applying that offset means knowing which element of P1["pose"] is the
# tool's yaw (rotation about the vertical axis a top-down suction pick
# actually needs). ROTATION_POSE_INDEX below assumes a 6-axis pose,
# [x, y, z, rx, ry, rz], with yaw last (index 5) -- confirmed correct for
# this Magician E6 by an actual pick-and-place cycle. The startup print
# below still shows the real P1["pose"] every run, in case that ever
# changes (a different taught point, a different robot).
#
# The direction/sign question (does a positive r rotate the same way as
# a positive orientation reading from the camera?) is also confirmed --
# it did NOT, by default, so the fix lives on the vision side:
# vision/orientation.py's angle_between() negates its result. Nothing
# here needed to change for that.

import time

buffer = ""

# Either spelling closes the connection -- matches whichever one the
# client actually sends (robot_client.py sends "CLOSE").
CLOSE_COMMANDS = ("CLOSE", "close_connection")

# Index into P1["pose"] that holds the tool's yaw -- see the comment
# above. Change this if your P1["pose"] isn't [x, y, z, rx, ry, rz].
ROTATION_POSE_INDEX = 5

def open_server():
    """Creates the listening socket and blocks until a client connects.
    Retries if the port isn't free yet (e.g. right after TCPDestroy)."""
    while True:
        err, sock = TCPCreate(True, "192.168.5.1", 6601)
        print("TCPCreate:", err)
        if err == 0:
            break
        time.sleep(1)
    print("Waiting for client...")
    TCPStart(sock, 0)
    print("Client connected")
    return sock

socket = open_server()

# Captured ONCE, before any POSE message can modify it -- every incoming
# r is added to THIS, never to whatever P1["pose"] happens to hold at the
# time (which would drift after repeated cycles). r=0 always means
# "exactly as taught", no matter how many pick cycles have run since.
print("P1 pose as taught:", P1["pose"])
BASE_ROTATION = P1["pose"][ROTATION_POSE_INDEX]

def send_reply(message):
    TCPWrite(socket, (message + "\n").encode("utf-8"))

while True:

    # Wait for data from Python client
    err, recBuf = TCPRead(socket)

    if err != 0 or not recBuf:
        # Client disconnected (a blocking read only returns empty at EOF).
        # This must NOT end the script -- only an explicit CLOSE does (see
        # header). The old handle can't accept a new client, so tear it
        # down and listen again; drop any half-received line with it.
        print("TCPRead error:", err, "-- client gone, re-listening")
        TCPDestroy(socket)
        buffer = ""
        socket = open_server()
        continue

    print("Received:", recBuf)

    # bytes -> string
    data = buffer + recBuf.decode("utf-8", errors="replace")
    messages = data.split("\n")
    buffer = messages.pop() if messages else ""

    should_close = False

    for message in messages:
        message = message.strip()

        if not message:
            continue

        print("Message:", message)

        # Ignore connection message
        if message == "Connection Established":
            print("Client connected")
            send_reply("READY")
            continue

        # Explicit close -- the ONLY thing that ends this script now.
        if message in CLOSE_COMMANDS:
            print("Close requested by client")
            send_reply("ACK:CLOSE")
            should_close = True
            break

        # Process POSE message
        if message.startswith("POSE,"):
            parts = message.split(",")

            if len(parts) != 5:
                print("Invalid POSE:", message)
                send_reply("ERR:INVALID_POSE")
                continue

            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                r = float(parts[4])

            except ValueError:
                print("Invalid coordinates:", message)
                send_reply("ERR:INVALID_COORDS")
                continue

            print("X =", x)
            print("Y =", y)
            print("Z =", z)
            print("r  =", r, "(offset from taught", BASE_ROTATION, ")")

            # Change P1. Rotation is BASE_ROTATION + r, never P1's current
            # value -- see BASE_ROTATION's own comment for why.
            P1["pose"][0] = x
            P1["pose"][1] = y
            P1["pose"][2] = z
            P1["pose"][ROTATION_POSE_INDEX] = BASE_ROTATION + r

            print("Moving P1:", P1["pose"])
            result = MovL(P1, {"user": 0, "v": 60})
            print("MovL result:", result)

            send_reply("ACK:POSE")
            continue

        # Process DO (digital output) message
        if message.startswith("DO,"):
            parts = message.split(",")

            if len(parts) != 3:
                print("Invalid DO:", message)
                send_reply("ERR:INVALID_DO")
                continue

            try:
                do_index = int(parts[1])
                do_state = ON if int(parts[2]) == 1 else OFF
            except ValueError:
                print("Invalid DO params:", message)
                send_reply("ERR:INVALID_DO_PARAMS")
                continue

            print(f"Setting DO_{do_index} = {do_state}")
            result = ToolDO(do_index, do_state)
            print("DO result:", result)

            send_reply("ACK:DO")
            continue

        print("Unknown message:", message)
        send_reply("ERR:UNKNOWN_COMMAND")

    if should_close:
        break

TCPDestroy(socket)
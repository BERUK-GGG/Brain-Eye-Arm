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
# ONE OPEN QUESTION THIS SCRIPT CAN'T RESOLVE ON ITS OWN
# --------------------------------------------------------
# Dobot doesn't document what TCPRead's error code actually means --
# specifically, whether `err != 0` can mean "no data yet, totally
# normal, keep waiting" (in which case looping back is obviously
# correct), or whether it can mean "this socket handle is now dead"
# (in which case looping back just spins on the same dead handle
# forever, and a NEW client would never be able to connect again
# without this script re-running TCPCreate/TCPStart from scratch).
# Watch this behavior the first time a client disconnects mid-session:
# if a new client can reconnect and get a "READY" reply afterward, this
# is fine as written. If a new client can't get through after a
# disconnect (you just see the same TCPRead error repeating forever),
# this loop needs to re-create the socket on error instead of just
# continuing -- tell me and I'll change it to do that.

import time

buffer = ""

# Either spelling closes the connection -- matches whichever one the
# client actually sends (robot_client.py sends "CLOSE").
CLOSE_COMMANDS = ("CLOSE", "close_connection")

err, socket = TCPCreate(True, "192.168.5.1", 6601)
print("TCPCreate:", err)

TCPStart(socket, 0)

def send_reply(message):
    TCPWrite(socket, (message + "\n").encode("utf-8"))

last_read_err = 0

while True:

    # Wait for data from Python client
    err, recBuf = TCPRead(socket)

    if err != 0:
        # A read error (e.g. client disconnected) must NOT end the script --
        # only an explicit CLOSE does (see header). Log once per run of
        # identical errors so a dead handle doesn't flood the console,
        # drop any half-received line from the old client, and back off
        # briefly instead of spinning.
        if err != last_read_err:
            print("TCPRead error:", err)
            last_read_err = err
        buffer = ""
        time.sleep(0.1)
        continue

    last_read_err = 0

    if not recBuf:
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

            if len(parts) != 4:
                print("Invalid POSE:", message)
                send_reply("ERR:INVALID_POSE")
                continue

            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])

            except ValueError:
                print("Invalid coordinates:", message)
                send_reply("ERR:INVALID_COORDS")
                continue

            print("X =", x)
            print("Y =", y)
            print("Z =", z)

            # Change P1
            P1["pose"][0] = x
            P1["pose"][1] = y
            P1["pose"][2] = z

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
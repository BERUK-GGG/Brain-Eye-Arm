import socket
import time

from detector import detect_objects, load_image

HOST = "192.10.190.234"
PORT = 7001
TIMEOUT = 5

MESSAGE = "Connection Established"
IMAGE_PATH = "imgs/test_20260915_174359_958859.jpg"

XY_HOME = ["221", "-130"]
Z_home = "150"
Z_safe = "90"
Z_pick = "18"
Z_place = "23"

TRIANGLE_PLACE = ["177.5687", "-267.7427"]
Circle_HOME = ["0", "0"]

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(TIMEOUT)


def recv_line():
    data = b""

    while True:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Server closed the connection")

        data += chunk
        if b"\n" in data:
            line, _, rest = data.partition(b"\n")
            # keep any extra bytes for the next call
            sock.setblocking(False)
            try:
                extra = sock.recv(4096)
                if extra:
                    data = extra + rest
                else:
                    data = rest
            except socket.error:
                data = rest
            finally:
                sock.settimeout(TIMEOUT)
            return line.decode("utf-8", errors="replace").strip()


def send_to_target(target):
    message = f"POSE,{target[0]},{target[1]},{Z_safe}\n"
    print(f"Sending: {message.strip()}")
    sock.sendall(message.encode("utf-8"))

    response = recv_line()
    print(f"Server reply: {response}")
    return response

def pick_target(target):
    message = f"POSE,{target[0]},{target[1]},{Z_pick}\n"
    print(f"Sending: {message.strip()}")
    sock.sendall(message.encode("utf-8"))

    response = recv_line()
    print(f"Server reply: {response}")
    return response

def place_target(shape):
    if shape == "Triangle":
        message = f"POSE,{TRIANGLE_PLACE[0]},{TRIANGLE_PLACE[1]},{Z_place}\n"
        print(f"Sending: {message.strip()}")
        sock.sendall(message.encode("utf-8"))

        response = recv_line()
        print(f"Server reply: {response}")
        return response

def send_home(target):
    message = f"POSE,{target[0]},{target[1]},{Z_home}\n"
    print(f"Sending: {message.strip()}")
    sock.sendall(message.encode("utf-8"))

    response = recv_line()
    print(f"Server reply: {response}")
    return response

def set_do(index, state):
    message = f"DO,{index},{state}\n"
    print(f"Sending: {message.strip()}")
    sock.sendall(message.encode("utf-8"))

    response = recv_line()
    print(f"Server reply: {response}")
    return response

def detect_targets():
    image = load_image(IMAGE_PATH)
    _, detections = detect_objects(image)

    # Make pick order repeatable because contour order is not guaranteed.
    detections.sort(key=lambda detection: detection["pixel"])
    return detections


def main():
    try:
        detections = detect_targets()
        if not detections:
            raise RuntimeError("No valid objects detected; robot connection not opened")

        print(f"Connecting to {HOST}:{PORT}...")
        sock.connect((HOST, PORT))
        print("Connected")

        # Tell server we're connected
        sock.sendall((MESSAGE + "\n").encode("utf-8"))
        print(f"Server says: {recv_line()}")

        for detection in detections:
            print(
                f"Detected {detection['shape']} at {detection['robot']} "
                f"with orientation {detection['orientation']:.2f} degrees"
            )
            send_to_target(detection["robot"])
            pick_target(detection["robot"])
            set_do(1, 1)
            time.sleep(0.3)
            send_to_target(detection["robot"])
            place_target(detection["shape"])
            time.sleep(0.3)
            set_do(1, 0)
            send_home(XY_HOME)

    except socket.timeout:
        print(f"ERROR: timeout after {TIMEOUT}s")

    except socket.error as e:
        print(f"ERROR: {e}")

    except (FileNotFoundError, RuntimeError) as e:
        print(f"ERROR: detection error: {e}")

    finally:
        sock.close()


if __name__ == "__main__":
    main()

# Brain-Eye-Arm

INA700 project: a Basler camera and a YOLO model on a Raspberry Pi 4 find red shapes (Triangle, Circle, Square). A Dobot Magician E6 picks each one with suction and places it in its slot on the board.

## How it works

```
Basler camera -> VisionService (grab + detect threads) -> Flask UI
                                                             | Pick button
                        PickAndPlaceController -> RobotClient --TCP 192.168.5.1:6601--> server_script/dobot_tcp_server.py (on the Dobot)
```

```
main.py          entry point: builds the objects below and starts the web app
config.py        loads config.json into typed settings

vision/
  camera.py        Camera: Basler grab
  detector.py      Detector: YOLO -> shape, orientation, color, robot coords
  calibration.py   Calibration: pixel -> robot x,y (homography)
  color.py         ColorClassifier: hue -> color name
  orientation.py   tilt-from-level: how far the object's topmost two corners are from level
  shapes.py        DetectedShape + one subclass per shape (self-registering)
  tracker.py       CentroidTracker: stable IDs across passes

robot/
  client.py        RobotClient: TCP link and commands (persistent connection)
  picker.py        PickAndPlaceController: the pick/place sequence

app/
  service.py       VisionService: camera + detection threads, shared results
  overlay.py       draws detections on a frame
  web.py           Flask routes: /, /video, /pick

server_script/dobot_tcp_server.py   runs inside DobotStudio Pro
```

Detector, calibration, color classifier and robot are passed in through constructors, so each can be swapped or faked in a test.

## Setup

```
pip install -r requirement.txt
```

Robot side:
1. In DobotStudio Pro, teach a point named `P1`.
2. Paste `server_script/dobot_tcp_server.py` into an Application script and run it.

## Run

```
python main.py
```

Open `http://<pi-ip>:5000`. The page shows the live feed with detections. **Pick & Place** picks every tracked object.

`python dobot_tcp_client.py` does a one-shot detect-and-pick to test the robot link. Stop `main.py` first, because the camera allows only one connection.

## Config (`config.json`)

- `model_path`, `yolo_conf_threshold`: the trained model and its confidence cutoff.
- `class_aliases`: model label to shape name, e.g. `"Rectangle": "Square"`.
- `calibration`: matching `image_points` and `robot_points`. Redo this whenever the camera or table moves.
- `place_targets`: robot x,y where each shape is dropped. **Only shapes listed here are detected**, so this is how the two unassigned shapes are ignored. A shape set to `null` is detected but not picked.
- `target_tilt`: per-shape "already correct" tilt in degrees (see "Orientation, precisely" below). Defaults to 0 for any shape not listed.
- `hue_ranges`: optional overrides for color names (OpenCV hue, 0-179).
- `robot`: host, port, timeout, home position, `z_safe`, `z_pick`, suction output index and settle time.


## Robot protocol

Text lines over TCP, and each message gets one reply.

| Send | Reply |
|---|---|
| `Connection Established` | `READY` |
| `POSE,x,y,z,r` | `ACK:POSE` |
| `DO,index,0/1` | `ACK:DO` |
| `CLOSE` | `ACK:CLOSE` (ends the robot script) |

`r` is a rotation **offset in degrees from however `P1` was taught**, not an absolute angle -- the server adds it to `P1`'s original rotation (captured once at startup), so `r=0` always means "exactly as taught," and it never drifts across repeated cycles.

Errors come back as `ERR:...` and raise `RobotCommandError`.

## Training

`models/industrial_ai_model_train.ipynb` trains the YOLO model. `scripts/capture_img.py` captures training images.

## Testing against a saved image

```
python scripts/detect_and_orient.py imgs/some_photo.png
```

Runs the real pipeline (same `Detector` main.py uses) on one saved image, no camera or robot needed. Prints one line per detection and saves an annotated copy (`<image>_annotated.png`) using the same overlay the live GUI draws, so it's a faithful preview of what `/video` would show.

## Orientation, precisely

`orientation` is **not** a compass angle -- it's how far (in degrees) the object's current topmost two corners are from level, wrapped to `(-90, 90]`. 0 means they're already level. See `vision/orientation.py`'s module docstring for the full reasoning.

This is confirmed for **Square** (axis-aligned reads ~0°, matching intuition). For **Triangle**, "level" under this definition means an *edge* is on top (apex hanging down) -- a triangle sitting apex-up, which is how it naturally sits on the table, reads a large tilt (~60°) instead of 0. Whether that matches what your physical slot needs hasn't been confirmed against the real board yet -- if not, set that shape's real value in `config.json`'s `target_tilt`.

The live GUI and `scripts/detect_and_orient.py`'s output both draw this directly: a yellow line between the two real corners used, a gray horizontal reference line showing "level", and the angle as text -- not just a number.

**Applying it:** `robot/picker.py` computes `correction = target_tilt[shape] - orientation` and rotates the tool by that many degrees while carrying the object -- pick happens at the tool's default rotation (a suction cup doesn't care about approach angle), then the move to the place point carries `r=correction`, and the final move home resets `r` to 0 for the next cycle. See `PickAndPlaceController._pick_and_place`.

## Known limits

- **Two things about the rotation above are unverified on real hardware** (can't be tested without the physical robot):
  1. Which element of `P1["pose"]` is the tool's yaw. `server_script/dobot_tcp_server.py` assumes index 5 (`[x, y, z, rx, ry, rz]`) and prints the real `P1["pose"]` at startup so this is easy to check.
  2. Whether a positive `orientation` reading from the camera and a positive `r` sent to the robot actually rotate the same direction. Test with one object at a known tilt; if it comes out backwards, negate `correction` in `robot/picker.py`.
- `/pick` picks everything tracked, with no selection by shape or color.
- Restart the robot script in DobotStudio Pro after any `CLOSE`.
- `models/yolo26n_best.onnx` is an experimental export sitting alongside the trained `.pt` file -- benchmarked faster (~2x on a test machine, not the Pi), but `config.json` still points at the `.pt` file; nothing currently uses the `.onnx` one.

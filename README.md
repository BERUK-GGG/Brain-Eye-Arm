# Brain-Eye-Arm

INA700 project: a Basler camera and a YOLO model on a Raspberry Pi 4 find shapes (Triangle, Circle, Square). A Dobot Magician E6 picks each one with suction and places it in its slot on the board.

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
  color.py         ColorClassifier: fixed palette (red, green, blue, black, white)
  orientation.py   tilt-from-level: how far the object's bottommost two corners are from level
  shapes.py        DetectedShape + one subclass per shape (self-registering)
  tracker.py       CentroidTracker: stable IDs across passes

robot/
  client.py        RobotClient: TCP link and commands (persistent connection)
  picker.py        PickAndPlaceController: the pick/place sequence

app/
  service.py       VisionService: camera + detection threads, shared results
  overlay.py       draws detections on a frame
  selection.py     filter_tracked: which detections match a shape/color pick request
  web.py           Flask routes for the HMI: /, /video, /detections, /status, /history, /shapes, /colors, /pick
  templates/, static/   the HMI page (index.html, hmi.css, hmi.js)

server_script/dobot_tcp_server.py   runs inside DobotStudio Pro

stand_alone_scripts/
  camera.py             grab one frame to imgs/ (for calibration)
  Calibration.py        click calibration points on that frame (needs a desktop GUI)
  robot_probe.py        test the TCP link step by step, no arm movement
  dobot_tcp_client.py   one-shot detect-and-pick, without the web app
  detect_and_orient.py  run detection on a saved image, no camera or robot
  capture_img.py        web-based capture of training images
```

Detector, calibration, color classifier and robot are passed in through constructors, so each can be swapped or faked in a test.

## Setup

Do these in order. Steps 4-6 have to be redone whenever the camera, the table or the tray moves.

### 1. Install

On the Pi, from the repo root:

```
pip install -r requirement.txt
```

`requirement.txt` uses `opencv-python-headless` (the Pi has no display). `stand_alone_scripts/Calibration.py` opens a window, so run it on a PC with the regular `opencv-python` instead.

### 2. Networking: put the Pi on the Dobot's subnet

The Dobot's LAN port has the fixed address **`192.168.5.1`** (subnet `192.168.5.0/24`). Anything that talks to it -- the PC running DobotStudio Pro, and the Pi -- needs a static address in that subnet. Pick any free one other than `.1`, e.g. `192.168.5.10` for the Pi and `192.168.5.20` for the PC.

**Raspberry Pi OS Bookworm (NetworkManager):**

```
nmcli con show                      # find the wired connection's name, usually "Wired connection 1"
sudo nmcli con mod "Wired connection 1" ipv4.method manual ipv4.addresses 192.168.5.10/24 ipv4.gateway ""
sudo nmcli con up "Wired connection 1"
```

**Older Raspberry Pi OS (dhcpcd)** -- add to `/etc/dhcpcd.conf`, then reboot:

```
interface eth0
static ip_address=192.168.5.10/24
```

Leave the gateway empty on the Ethernet side. The Dobot isn't a router, and a gateway there would steal the Pi's default route from Wi-Fi, which is how you reach the web UI.

**Windows PC (for DobotStudio Pro):** Settings -> Network -> Ethernet -> IP assignment -> Manual, IPv4 `192.168.5.20`, subnet mask `255.255.255.0`, no gateway.

**Check it:**

```
ping 192.168.5.1
python -m stand_alone_scripts.robot_probe        # once the robot script is running (step 3)
```

Ping only proves the network. `robot_probe` proves the robot script is listening and answering -- see Troubleshooting for what each failed step means.

Tip: the Dobot has one LAN port, so the usual routine is: start the script from the PC, then move the cable to the Pi. A small Ethernet switch lets the PC and the Pi stay connected at the same time.

### 3. Robot side

1. In DobotStudio Pro, enable the robot and clear any alarms.
2. Teach a point named **`P1`**: tool pointing straight down, wrist at the rotation you want as "default". Every `r` the Pi sends is an offset from P1's rotation (see Robot protocol). **The script crashes without `P1`**, and the Pi then sees `connection reset by peer` on the handshake.
3. Paste `server_script/dobot_tcp_server.py` into an Application script and run it. Its console should print `Waiting for client...`, then `Client connected` and `P1 pose as taught: [...]` once the Pi connects.
4. The suction cup is on tool digital output 1 (`robot.suction_do_index` in `config.json`).

The script keeps running across Pi reconnects. Only a `CLOSE` message ends it. `main.py` does not send `CLOSE` on shutdown, but `stand_alone_scripts/dobot_tcp_client.py` does, so restart the script in DobotStudio Pro after using that one.

### 4. Calibrate camera pixels to robot coordinates

`vision/calibration.py` fits a homography from matching point pairs: where a spot is in the image, and where the robot's tool is when it sits on that spot. Use 5 points (4 is the minimum; 5+ gets a least-squares fit that evens out jogging errors).

1. **Mount the camera for good.** Any change in camera position, zoom or focus invalidates the calibration.
2. **Place 5 markers** on the table, spread over the whole area where objects will lie: the four corners plus the centre works well. No three in a line. Small crosses or stickers with a clear centre are easiest to hit precisely.
3. **Take a photo** with the Pi (stop `main.py` first -- the camera allows one connection):
   ```
   python stand_alone_scripts/camera.py      # saves imgs/test_<timestamp>.jpg
   ```
   This is the camera's full resolution, the same frames `main.py` detects on.
4. **Record robot coordinates.** In DobotStudio Pro, jog the suction tip onto each marker's centre and note X and Y (base / user frame 0 -- the same frame the script's `MovL` uses). Keep the order; you'll click them in the same order.
5. **Click the points.** Copy the photo to the PC. In `stand_alone_scripts/Calibration.py`, set `image_path` to the photo and `robot_points` to your 5 recorded X,Y pairs, then:
   ```
   python stand_alone_scripts/Calibration.py
   ```
   Click each marker's centre in the same order as `robot_points`, then press Esc. Clicks are scaled back to full-resolution pixels for you.
6. **Paste** the printed `"calibration"` block into `config.json`.
7. **Verify.** Put an object somewhere new, run `python stand_alone_scripts/detect_and_orient.py <photo>` (or watch `main.py`'s console), and compare the printed `robot=(x, y)` with the X,Y you get by jogging onto the object. A few mm is fine. Larger errors usually mean a point was clicked out of order.

If startup fails with `Camera->robot calibration failed`, two points are duplicated or three are collinear.

### 5. Place targets and heights

All in `config.json`, all in robot coordinates read from DobotStudio Pro by jogging:

- `place_targets`: X,Y of each slot's centre on the tray.
- `robot.z_pick`: Z where the suction cup just presses on an object's top. Also used for placing.
- `robot.z_safe`: travel height, clear of everything on the table.
- `robot.home_xy`: where the arm waits between cycles. Keep it out of the camera's view.

### 6. Slot orientation (`target_tilt`)

The rotation each shape needs in its slot, in the same units the GUI shows (see "Orientation, precisely"). Circles don't need one. For a square, 0 is right for a slot aligned with the camera.

To set it for a triangle (or a rotated square slot): put the shape under the camera turned exactly as the slot requires, read the angle the GUI shows, and put that number in `target_tilt`. The camera measures both the slot pose and the picked object the same way, so the camera-to-robot angle cancels out and needs no extra calibration.

Triangle reference values (left/right as seen **in the camera image**, which is upside-down compared to the table):

| Slot's point faces (in the image) | `target_tilt.Triangle` |
|---|---|
| Up | 0 |
| Down | 60 (-60 is the same pose) |
| Left | -30 |
| Right | 30 |

When in doubt, measure rather than use the table.

## Run

```
python main.py
```

Open `http://<pi-ip>:5000` (the Pi's Wi-Fi address). The operator HMI has five areas:

- **Status bar**: camera (live / stalled), vision (seconds per detection pass), robot link (idle, connected, running a cycle) and a clock.
- **Live camera**: the feed with each part's ID, color, shape, confidence and measured orientation edge drawn on it.
- **Pick command**: shape and color dropdowns (shapes from `place_targets`, colors from the fixed palette in `vision/color.py`). The button picks every tracked part that matches, and shows how many that is. The outcome of the last pick is shown below it.
- **Detected parts**: one row per tracked part (ID, shape, color, confidence, angle, robot x/y), with a **Pick** button to pick just that part. Rows matching the current selection are highlighted.
- **Pick history**: one row per handled part, newest first: time, ID, color and shape, where it was picked, its measured angle, how far it was rotated, where it was placed, and the result (placed / skipped / failed, with the error). Totals are shown in the header, and **Export CSV** downloads the table. The history covers the current run of `main.py` only.

The buttons are locked while a cycle runs, from any browser.

API, for scripting:

| Route | |
|---|---|
| `POST /pick` | Optional JSON body: `{"shape": "Triangle", "color": "red"}` (either field, or none, matches anything) or `{"object_id": 3}` for one part. |
| `GET /detections` | Tracked parts. |
| `GET /status` | Camera/vision timings, robot link state, last pick outcome. |
| `GET /history?after=N` | Handled parts (and failed requests) numbered after `N`. |
| `GET /shapes`, `GET /colors` | What the dropdowns are built from. |

`python -m stand_alone_scripts.dobot_tcp_client` does a one-shot detect-and-pick to test the robot link. Stop `main.py` first, because the camera allows only one connection. It sends `CLOSE` at the end, so restart the robot script afterwards.

## Troubleshooting

`python -m stand_alone_scripts.robot_probe` tests connect, handshake and a suction-off command one at a time (`--move` adds one move to home at `z_safe`).

| Symptom | Likely cause |
|---|---|
| Ping fails | Pi or PC not on `192.168.5.x/24`, or cable not plugged in. See step 2. |
| Probe fails at `connect` (refused / timeout) | Robot script isn't running. Start it in DobotStudio Pro. |
| Probe fails at `handshake` with `connection reset` | Robot script crashed right after accepting -- most often `P1` not taught. Check the DobotStudio Pro console. |
| Probe fails at `DO` or `POSE` | The command crashed the script: robot not enabled, in alarm, or the pose is unreachable. |
| `/pick` returns 500 with `[Errno 104]` | Same as the `handshake` row. |
| Object picked at the wrong spot | Recalibrate (step 4). |
| Object placed rotated wrong | Check `target_tilt` (step 6). |

## Config (`config.json`)

- `model_path`, `yolo_conf_threshold`: the trained model and its confidence cutoff. The `.onnx` export is much faster on the Pi than the `.pt` (~300 ms vs ~2 s per frame).
- `class_aliases`: model label to shape name, e.g. `"Rectangle": "Square"`.
- `calibration`: matching `image_points` and `robot_points`. See setup step 4.
- `place_targets`: robot x,y where each shape is dropped. **Only shapes listed here are detected**, so this is how the two unassigned shapes are ignored. A shape set to `null` is detected but not picked.
- `target_tilt`: per-shape slot rotation in degrees. See setup step 6. Defaults to 0 for any shape not listed.
- `color_thresholds`: optional overrides for the color classifier, e.g. `{"black_max": 0.45, "white_max_chroma": 0.3}`. See `vision/color.py`.
- `robot`: host, port, timeout, home position, `z_safe`, `z_pick`, suction output index and settle time, and fail recovery: `pick_retries` (default 2) extra attempts when a part is still lying at its pick spot after a cycle, each `retry_z_step` mm (default 1) lower than the last.

## Robot protocol

Text lines over TCP, and each message gets one reply.

| Send | Reply |
|---|---|
| `Connection Established` | `READY` |
| `POSE,x,y,z,r` | `ACK:POSE` |
| `DO,index,0/1` | `ACK:DO` |
| `CLOSE` | `ACK:CLOSE` (ends the robot script) |

`r` is a rotation **offset in degrees from however `P1` was taught**, not an absolute angle -- the server adds it to `P1`'s original rotation (captured once at startup), so `r=0` always means "exactly as taught," and it never drifts across repeated cycles.

Errors come back as `ERR:...` and raise `RobotCommandError`. When a client disconnects, the script goes back to listening for a new one. The Pi checks a kept-open connection with the handshake before each pick and reconnects if it went stale.

## Training

`models/industrial_ai_model_train.ipynb` trains the YOLO model. `stand_alone_scripts/capture_img.py` captures training images through a browser.

## Testing against a saved image

```
python stand_alone_scripts/detect_and_orient.py imgs/some_photo.png
```

Runs the real pipeline (same `Detector` main.py uses) on one saved image, no camera or robot needed. Prints one line per detection, including its robot x,y, and saves an annotated copy (`<image>_annotated.png`) using the same overlay the live GUI draws, so it's a faithful preview of what `/video` would show.

## Orientation, precisely

`orientation` is **not** a compass angle -- it's how far (in degrees) the object's current bottommost two corners are from level, wrapped to `(-90, 90]`. 0 means they're already level. Bottommost, not topmost, because the camera's image is upside-down relative to the physical scene -- the pair that actually corresponds to the object's real top edge is its bottommost two corners in the (inverted) pixel frame. See `vision/orientation.py`'s module docstring for the full reasoning.

An axis-aligned **Square** reads ~0°, and a **Triangle** sitting apex-up also reads ~0°, since its base is then the level pair.

The live GUI and `stand_alone_scripts/detect_and_orient.py`'s output both draw this directly: a yellow line between the two real corners used, a gray horizontal reference line showing "level", and the angle as text -- not just a number.

**Applying it:** `robot/picker.py` computes `correction = target_tilt[shape] - orientation` and rotates the tool by that many degrees while carrying the object -- pick happens at the tool's default rotation (a suction cup doesn't care about approach angle), then the move to the place point carries `r=correction`, and the final move home resets `r` to 0 for the next cycle. See `PickAndPlaceController._pick_and_place`.

**Symmetry:** a triangle looks the same every 120° and a square every 90° (`SYMMETRY_DEG` in `vision/shapes.py`). The correction is wrapped into ±60° / ±45°, so the wrist never turns further than needed, whatever angle the slot sits at.

**Sign convention -- confirmed on hardware:** a plain image-frame angle rotated the robot the opposite way from what was measured. `vision/orientation.py`'s `angle_between` negates its result to correct for this; that negation (not `robot/picker.py`) is where to look if this ever needs revisiting (e.g. after a camera remount).

## Known limits

- Restart the robot script in DobotStudio Pro after any `CLOSE`.
- The Dobot has one LAN port that accepts outside connection (LAN 1), so the ethernet cable has to move from the PC to the Pi after starting the server-script.

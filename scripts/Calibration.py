import cv2
import json

image_path = "basler_20260930_163914.png"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not open {image_path}")

original_height, original_width = image.shape[:2]

print("Original image size:", original_width, "x", original_height)
print("Click the 5 calibration points.")
print("Press ESC when finished.")

# Maximum display size
max_width = 1200
max_height = 1000

# Calculate scale while keeping the entire image visible
scale = min(
    max_width / original_width,
    max_height / original_height,
    1.0
)

display_width = int(original_width * scale)
display_height = int(original_height * scale)

display_image = cv2.resize(
    image,
    (display_width, display_height),
    interpolation=cv2.INTER_AREA
)

print(f"Display size: {display_width} x {display_height}")
print(f"Display scale: {scale:.4f}")

image_points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:

        # Convert displayed coordinates to original image coordinates
        original_x = int(round(x / scale))
        original_y = int(round(y / scale))

        image_points.append([original_x, original_y])

        print(
            f"Point {len(image_points)}: "
            f"[{original_x}, {original_y}]"
        )


cv2.namedWindow("Calibration Image", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Calibration Image", mouse_callback)

while True:
    cv2.imshow("Calibration Image", display_image)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

cv2.destroyAllWindows()


# --------------------------------------------------
# Robot coordinates
# --------------------------------------------------

robot_points = [
    [-126.0168, -375.1351],
    [-127.1461, -236.3346],
    [-47.2314, -302.4081],
    [28.0944, -366.4028],
    [30.6247, -226.1670]
]


# --------------------------------------------------
# Check that the number of points matches
# --------------------------------------------------

if len(image_points) != len(robot_points):
    raise ValueError(
        f"Expected {len(robot_points)} image points, "
        f"but got {len(image_points)}."
    )


# --------------------------------------------------
# Create calibration JSON
# --------------------------------------------------

calibration = {
    "calibration": {
        "image_points": image_points,
        "robot_points": robot_points
    }
}


print("\nCalibration data:")
print(json.dumps(calibration, indent=2))

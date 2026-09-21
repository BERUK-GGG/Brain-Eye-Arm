from pypylon import pylon
import cv2
import os
from datetime import datetime

# Connect to the first available camera
camera = pylon.InstantCamera(
    pylon.TlFactory.GetInstance().CreateFirstDevice()
)

# Start grabbing one image
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# Convert Basler image to OpenCV BGR format
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# Folder where photos will be saved
save_folder = "imgs"
os.makedirs(save_folder, exist_ok=True)

print("Camera started. Capturing image...")

try:
    # Grab exactly one image
    grabResult = camera.RetrieveResult(
        5000,
        pylon.TimeoutHandling_ThrowException
    )

    try:
        if grabResult.GrabSucceeded():

            # Convert image to OpenCV format
            image = converter.Convert(grabResult)
            img = image.GetArray()

            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(
                save_folder,
                "test_{}.jpg".format(timestamp)
            )

            # Save image
            cv2.imwrite(filename, img)

            print("Photo saved:", filename)

        else:
            print("Failed to grab image.")

    finally:
        grabResult.Release()

finally:
    # Stop camera
    camera.StopGrabbing()
    print("Camera stopped.")

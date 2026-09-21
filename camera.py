"""
Shared Basler (pypylon) camera setup + single-frame grab.

Both stream.py (continuous grab_loop, many frames) and
dobot_tcp_client.py (one-shot grab at pick time) need the exact same
camera setup -- InstantCamera + ImageFormatConverter construction, and
the retrieve/convert/release sequence for pulling one frame. That setup
used to be duplicated almost line-for-line in both files, which meant a
change to camera config (pixel format, bit alignment, timeout, etc.)
had to be remembered and made twice. It lives here once instead.
"""

from pypylon import pylon


def open_camera():
    """Opens the Basler camera and starts grabbing. Returns (camera,
    converter) -- callers own both and are responsible for calling
    close_camera() when done with them."""
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    return camera, converter


def grab_frame(camera, converter, timeout_ms=5000):
    """Retrieves and converts one frame. Returns a BGR numpy array, or
    None if this particular grab didn't succeed (caller decides whether
    to retry -- stream.py's grab_loop just tries again next iteration;
    dobot_tcp_client.py's one-shot grab treats it as an error)."""
    grab_result = camera.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
    try:
        if not grab_result.GrabSucceeded():
            return None
        image = converter.Convert(grab_result)
        return image.GetArray()
    finally:
        grab_result.Release()


def close_camera(camera):
    camera.StopGrabbing()
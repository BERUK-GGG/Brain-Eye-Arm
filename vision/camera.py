"""Basler (pypylon) camera wrapper, shared by the live app and the one-shot test client."""

from pypylon import pylon


class Camera:
    """Owns one Basler camera and its BGR converter. Use as a context
    manager, or call open()/close() yourself."""

    def __init__(self, timeout_ms: int = 5000):
        self._timeout_ms = timeout_ms
        self._camera = None
        self._converter = None

    def open(self) -> "Camera":
        self._camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        return self

    @property
    def is_grabbing(self) -> bool:
        return self._camera is not None and self._camera.IsGrabbing()

    def grab(self):
        """Returns one BGR numpy frame, or None if this grab didn't
        succeed (the caller decides whether to retry)."""
        result = self._camera.RetrieveResult(self._timeout_ms, pylon.TimeoutHandling_ThrowException)
        try:
            if not result.GrabSucceeded():
                return None
            return self._converter.Convert(result).GetArray()
        finally:
            result.Release()

    def close(self) -> None:
        if self._camera is not None:
            self._camera.StopGrabbing()
            self._camera = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

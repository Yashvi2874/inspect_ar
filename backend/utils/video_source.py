"""
Open a video source that may be a local camera, a phone, or a file.

At the hackathon we ran the pipeline off a phone camera by pointing OpenCV at an
IP-camera app's MJPEG stream instead of the laptop webcam. That was a one-line
edit at the time and never made it into the code, so the repository only ever
opened `cv2.VideoCapture(0)`. This module makes it a first-class option.

Why a phone is worth supporting: a laptop webcam is fixed, low-resolution and
attached to the wrong object. A phone can be walked around an airframe, held at
an angle, or eventually mounted on a UAV, which is the direction this project is
headed.
"""

import os
from typing import Union

import cv2

# Read by open_capture() when no source is passed, so the whole application can
# be pointed at a phone without editing code:
#     set INSPECT_AR_SOURCE=http://192.168.1.5:8080/video
ENV_VAR = "INSPECT_AR_SOURCE"

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720


def resolve_source(source: Union[int, str, None] = None) -> Union[int, str]:
    """
    Work out what to hand to cv2.VideoCapture.

    Accepts, in order of precedence: an explicit argument, then the
    INSPECT_AR_SOURCE environment variable, then the default webcam.

    A string of digits becomes an int, because OpenCV treats "0" (a filename)
    and 0 (a device index) as completely different things, and passing the
    string silently fails to open the camera.
    """
    if source is None:
        source = os.environ.get(ENV_VAR, 0)

    if isinstance(source, str):
        stripped = source.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped

    return source


def describe(source: Union[int, str]) -> str:
    """A human-readable name for the source, for log lines."""
    if isinstance(source, int):
        return f"local camera {source}"
    if source.startswith(("http://", "https://", "rtsp://")):
        return f"network stream {source}"
    return f"file {source}"


def open_capture(
    source: Union[int, str, None] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> cv2.VideoCapture:
    """
    Open a capture and return it, or raise with a message that says what to fix.

    Args:
        source: device index, stream URL, or video file. See resolve_source.
        width, height: requested frame size. Ignored by most network streams,
            which serve whatever the phone is configured to send.

    Raises:
        RuntimeError: the source could not be opened.
    """
    resolved = resolve_source(source)
    print(f"Opening {describe(resolved)} ...")

    capture = cv2.VideoCapture(resolved)

    if not capture.isOpened():
        if isinstance(resolved, int):
            raise RuntimeError(
                f"Could not open local camera {resolved}. Another application "
                "may be holding it, or there may be no camera attached."
            )
        raise RuntimeError(
            f"Could not open {resolved}. Check that the phone and this machine "
            "are on the same network, that the IP camera app is running, and "
            "that the URL opens in a browser. Most such apps expose a stream "
            "at a path like /video or /videofeed."
        )

    # Only meaningful for local cameras; network streams ignore it silently.
    if isinstance(resolved, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # A capture can report isOpened() and still fail on the first read when a
    # URL points at something that is not a video stream, so prove it works.
    ok, _ = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError(
            f"Opened {describe(resolved)} but could not read a frame from it. "
            "If this is a phone, confirm the stream URL is the video endpoint "
            "and not the app's web page."
        )

    return capture

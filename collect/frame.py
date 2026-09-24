"""Frame geometry shared by collection, training and inference.

The KING 5 Queen Anne tower camera burns a timestamp/branding strip into the
bottom ~25 px of every 1920x1080 frame ("9:49:37 AM - 9/23/2026 - Queen Anne
Tower Camera - KING 5", white on black). It is the most consistently
high-contrast thing in the picture and it changes every frame, so a classifier
is free to learn the clock instead of the mountain. Crop it before the model
ever sees it.

The crop happens at LOAD time, not at capture time: `collect/collector.py`
archives the raw frame, so the strip is still there if it is ever needed (it is
the only in-band record of when the camera says the frame was taken), and a
change to this number does not invalidate the capture archive.

Sizing: 25 px of 1080 is ~2.3% of the height, which survives `Resize(224)` as
about 5 px — small, but 5 px of flickering white text at the bottom of a
224x224 crop is plenty for a network to key on.
"""

from __future__ import annotations

#: Default strip height, in pixels of the source frame. Overridable per camera
#: via `[webcam] crop_bottom_px` in mountain.toml.
DEFAULT_CROP_BOTTOM_PX = 25


def crop_burn_in(frame, pixels: int = DEFAULT_CROP_BOTTOM_PX):
    """Return *frame* with its bottom *pixels* rows removed.

    Accepts a numpy array shaped (H, W, C) — what cv2 hands back — or a PIL
    Image, and returns the same type. A non-positive *pixels*, or a frame too
    short to crop, is a no-op: this is called on every training sample and on
    every tick, and it must never be the thing that raises.
    """
    if pixels <= 0:
        return frame

    # PIL Image (duck-typed: it has .crop and .size, a numpy array has neither)
    if hasattr(frame, "crop") and hasattr(frame, "size"):
        width, height = frame.size
        if height <= pixels:
            return frame
        return frame.crop((0, 0, width, height - pixels))

    height = frame.shape[0]
    if height <= pixels:
        return frame
    return frame[: height - pixels]


class CropBurnIn:
    """`crop_burn_in` as a torchvision-style transform.

    Goes at the HEAD of a Compose, before `ToPILImage`/`Resize`, so the strip
    is gone before any scaling folds it into neighbouring rows.
    """

    def __init__(self, pixels: int = DEFAULT_CROP_BOTTOM_PX):
        self.pixels = pixels

    def __call__(self, frame):
        return crop_burn_in(frame, self.pixels)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(pixels={self.pixels})"

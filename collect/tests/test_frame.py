import numpy as np
from PIL import Image

from collect.frame import DEFAULT_CROP_BOTTOM_PX, CropBurnIn, crop_burn_in


def test_crops_bottom_rows_from_numpy_frame():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out = crop_burn_in(frame, 25)
    assert out.shape == (1055, 1920, 3)


def test_crops_bottom_rows_from_pil_image():
    img = Image.new("RGB", (1920, 1080))
    out = crop_burn_in(img, 25)
    assert out.size == (1920, 1055)


def test_drops_the_burn_in_strip_not_the_view():
    # Bottom 25 rows white (the station's clock), the rest black.
    frame = np.zeros((100, 40, 3), dtype=np.uint8)
    frame[75:] = 255
    out = crop_burn_in(frame, 25)
    assert out.shape[0] == 75
    assert out.max() == 0


def test_zero_or_negative_is_a_no_op():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert crop_burn_in(frame, 0).shape == (10, 10, 3)
    assert crop_burn_in(frame, -5).shape == (10, 10, 3)


def test_frame_shorter_than_the_strip_is_left_alone():
    # Never the thing that raises: it runs on every sample and every tick.
    frame = np.zeros((20, 10, 3), dtype=np.uint8)
    assert crop_burn_in(frame, 25).shape == (20, 10, 3)
    assert crop_burn_in(Image.new("RGB", (10, 20)), 25).size == (10, 20)


def test_transform_wrapper_matches_the_function():
    frame = np.zeros((100, 40, 3), dtype=np.uint8)
    assert CropBurnIn(25)(frame).shape == crop_burn_in(frame, 25).shape
    assert "25" in repr(CropBurnIn(25))
    assert DEFAULT_CROP_BOTTOM_PX == 25

import torch
import numpy as np
from unittest.mock import MagicMock, patch
from train.utils import WebcamStream


def test_mps_availability():
    """Verify code identifies if MPS is available or defaults to CPU."""
    stream = WebcamStream(0)
    if torch.backends.mps.is_available():
        assert stream.device == "mps"
    else:
        assert stream.device == "cpu"


@patch("cv2.VideoCapture")
def test_no_disk_usage(mock_video_capture):
    """
    Mock OpenCV frame capture and verify that the resulting tensor
    is on the correct device and matches expected dimensions.
    """
    mock_cap = MagicMock()
    # Create a random BGR frame (OpenCV format)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, frame)
    mock_video_capture.return_value = mock_cap

    stream = WebcamStream(0)

    # Track calls to cv2.imwrite to ensure no disk usage
    with patch("cv2.imwrite") as mock_imwrite:
        tensor = stream.capture_to_tensor()
        mock_imwrite.assert_not_called()

    assert tensor is not None
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    # Check device (CPU in tests typically, or MPS if runner has it)
    if torch.backends.mps.is_available():
        assert tensor.device.type == "mps"


@patch("cv2.VideoCapture")
def test_capture_failure(mock_video_capture):
    """Verify behavior when frame capture fails."""
    mock_cap = MagicMock()
    mock_cap.read.return_value = (False, None)
    mock_video_capture.return_value = mock_cap

    stream = WebcamStream(0)
    tensor = stream.capture_to_tensor()
    assert tensor is None

import os
from pathlib import Path

import cv2
import pytest
import torch
from torchvision import transforms

from train.config_loader import ConfigLoader
from train.model import ConvNextLoRAModel

# Trained weights are no longer committed (R2 `checkpoints/` is the source of
# truth — see CHECKPOINTS.md). These tests need a real checkpoint: a local one
# in train/checkpoints/ (dev machines, the mini), or R2 credentials to pull it.
_LOCAL_CHECKPOINT = Path("train/checkpoints/classifier.pt")
_HAS_R2_CREDS = bool(os.environ.get("R2_ACCESS_KEY_ID"))


class RegressionTester:
    def __init__(self, config_path="mountain.toml"):
        self.config = ConfigLoader(config_path)
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        storage = self.config.get_storage("data") if _HAS_R2_CREDS else None
        self.model = ConvNextLoRAModel(
            checkpoint_dir=self.config.checkpoint_dir, storage=storage
        ).to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict(self, img_path, vis=1.0, ceil=1.0):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        weather_tensor = torch.tensor([[vis, ceil]], dtype=torch.float32).to(
            self.device
        )

        with torch.no_grad():
            output = self.model(img_tensor, weather_tensor)
            prediction = torch.argmax(output, dim=1).item()
        return prediction


@pytest.fixture(scope="module")
def tester():
    if not _LOCAL_CHECKPOINT.exists() and not _HAS_R2_CREDS:
        pytest.skip(
            "regression tests need trained weights: no local train/checkpoints/ "
            "and no R2 credentials to pull them (source cf.env)"
        )
    return RegressionTester()


def test_dark_frame_prediction(tester):
    """
    Case: Confirmed dark night frame.
    Human Label: 0 (Not Out)
    Source: assets/regression_samples/dark_sample.jpg
    """
    img_path = Path("assets/regression_samples/dark_sample.jpg")
    # Low visibility/ceiling usually accompanies dark frames if weather is bad,
    # but for a 'pure darkness' test we keep weather neutral to test vision bias.
    prediction = tester.predict(img_path, vis=1.0, ceil=1.0)

    assert prediction == 0, (
        f"Model failed to identify dark frame as 'Not Out'. Predicted: {prediction}"
    )


def test_known_out_frame(tester):
    """
    Case: Confirmed mountain-VISIBLE frame (binary-era sample, 2026-02-22).

    This frame was hand-labeled "1 (Mountain is Out)" under the original BINARY
    taxonomy. The 2026-03-14 reclassification (CHECKPOINTS.md) split "Out" into
    Full/Partial, and this frame — heavy overcast with the range only peeking
    through a horizon strip — is a textbook Partial under the current labels;
    every 3-class checkpoint since has predicted 2 for it. The contract this
    sample actually encodes is "a known-visible frame must never be called
    Not Out", so assert visibility, not Full.
    """
    img_path = Path("assets/regression_samples/mountain_out_sample.jpg")
    prediction = tester.predict(img_path, vis=1.0, ceil=1.0)

    assert prediction in (1, 2), (
        f"Model failed to see the mountain in a known-visible frame. "
        f"Predicted: {prediction} (Not Out)"
    )

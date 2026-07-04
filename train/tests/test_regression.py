import pytest
import torch
import cv2
from pathlib import Path
from torchvision import transforms
from train.model import ConvNextLoRAModel
from train.config_loader import ConfigLoader


class RegressionTester:
    def __init__(self, config_path="mountain.toml"):
        self.config = ConfigLoader(config_path)
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model = ConvNextLoRAModel(checkpoint_dir=self.config.checkpoint_dir).to(
            self.device
        )
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
    Case: Confirmed mountain visible frame.
    Human Label: 1 (Mountain is Out)
    Source: assets/regression_samples/mountain_out_sample.jpg
    """
    img_path = Path("assets/regression_samples/mountain_out_sample.jpg")
    prediction = tester.predict(img_path, vis=1.0, ceil=1.0)

    assert prediction == 1, (
        f"Model failed to identify known mountain-out frame. Predicted: {prediction}"
    )

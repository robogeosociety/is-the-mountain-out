import logging
import cv2
import torch
import numpy as np
from torchvision import transforms
from typing import Optional, Union
import requests
from metar import Metar


class WebcamStream:
    def __init__(self, source: Union[int, str], device: str = "mps"):
        self.source = source
        self.device = device if torch.backends.mps.is_available() else "cpu"
        self.cap = cv2.VideoCapture(source)

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

    def capture_to_tensor(self) -> Optional[torch.Tensor]:
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(frame_rgb).to(self.device)
        return tensor.unsqueeze(0)

    def capture_raw(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

    def __del__(self):
        self.release()


class WeatherFetcher:
    def __init__(self, station_id: str = "KSEA"):
        self.station_id = station_id
        self.base_url = f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{station_id.upper()}.TXT"

    def fetch_latest_metar(self) -> Optional[str]:
        try:
            response = requests.get(self.base_url, timeout=10)
            if response.status_code == 200:
                lines = response.text.strip().split("\n")
                return lines[-1] if lines else None
        except requests.RequestException as e:
            logging.warning(f"Error fetching METAR: {e}")
        return None

    def parse_metar_to_vector(self, metar_text: str) -> torch.Tensor:
        vis, ceil = 0.0, 1.0
        if metar_text:
            try:
                obs = Metar.Metar(metar_text)
                if obs.vis:
                    vis = min(obs.vis.value("SM"), 10.0) / 10.0
                if obs.sky:
                    layers = [layer for layer in obs.sky if layer[0] in ["BKN", "OVC"]]
                    ceil = (
                        min(layers[0][1].value("FT"), 10000.0) / 10000.0
                        if layers
                        else 1.0
                    )
            except (Metar.ParserError, AttributeError, IndexError, ValueError) as e:
                # Live METAR text off the NOAA feed is sometimes truncated or
                # malformed (station outage, mid-write scrape, format drift).
                # Metar.Metar() wraps any internal parse failure as ParserError
                # in strict mode; AttributeError/IndexError/ValueError cover the
                # sky-layer/value() lookups above on a report that parsed but has
                # an unexpected shape. Fall back to the neutral defaults already
                # set above rather than taking down the training loop.
                logging.warning(f"Could not parse METAR text {metar_text!r}: {e}")
        return torch.tensor([vis, ceil], dtype=torch.float32)

    def get_weather_vector(self) -> torch.Tensor:
        metar_text = self.fetch_latest_metar()
        return self.parse_metar_to_vector(metar_text)

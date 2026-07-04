"""Single-shot inference: fetch webcam + METAR, write state.json.

Used by `.github/workflows/update.yml` on a 15-minute schedule.
"""

import argparse
import io
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import requests
import torch
import torch.nn.functional as F
from metar import Metar
from PIL import Image
from torchvision import transforms

# Make `train` importable when invoked as a module or a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.config_loader import ConfigLoader  # noqa: E402
from train.model import ConvNextLoRAModel  # noqa: E402

CLASS_NAMES = ["not_out", "full", "partial"]

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def fetch_webcam_tensor(url: str, timeout: float = 20.0) -> torch.Tensor:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return IMAGE_TRANSFORM(img).unsqueeze(0)


def fetch_metar(station: str, timeout: float = 10.0) -> tuple[torch.Tensor, dict]:
    """Return (model_input_vector, raw_readout) for display and inference."""
    url = f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{station.upper()}.TXT"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    lines = resp.text.strip().splitlines()
    metar_text = lines[-1] if lines else ""

    vis_raw: float | None = None
    ceil_raw: float | None = None
    try:
        obs = Metar.Metar(metar_text)
        if obs.vis:
            vis_raw = obs.vis.value("SM")
        if obs.sky:
            layers = [layer for layer in obs.sky if layer[0] in ("BKN", "OVC")]
            if layers:
                ceil_raw = layers[0][1].value("FT")
    except Exception:
        pass

    vis_norm = min(vis_raw, 10.0) / 10.0 if vis_raw is not None else 0.0
    ceil_norm = min(ceil_raw, 10000.0) / 10000.0 if ceil_raw is not None else 1.0
    vector = torch.tensor([[vis_norm, ceil_norm]], dtype=torch.float32)

    readout = {
        "station": station.upper(),
        "visibility_sm": vis_raw,
        "ceiling_ft": ceil_raw,
        "raw": metar_text,
    }
    return vector, readout


def git_short_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip() or None
    except Exception:
        return None


def predict(checkpoint_dir: str, webcam_url: str, station: str, storage=None) -> dict:
    model = ConvNextLoRAModel(
        num_classes=3, checkpoint_dir=checkpoint_dir, device="cpu", storage=storage
    )
    model.model_dict.eval()

    image_tensor = fetch_webcam_tensor(webcam_url)
    weather_tensor, weather_readout = fetch_metar(station)

    with torch.no_grad():
        logits = model(image_tensor, weather_tensor)
        probs = F.softmax(logits, dim=1)[0].tolist()
    idx = int(max(range(3), key=lambda i: probs[i]))

    return {
        "timestamp_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "class_index": idx,
        "class_name": CLASS_NAMES[idx],
        "is_out": idx in (1, 2),
        "confidence": {name: probs[i] for i, name in enumerate(CLASS_NAMES)},
        "weather": weather_readout,
        "webcam_url": webcam_url,
        "model_version": git_short_sha(),
    }


def _iso_utc(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def _append_log(log_path: Path, record: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a single visibility prediction and write state.json."
    )
    parser.add_argument("--config", default="mountain.toml")
    parser.add_argument("--out", default="web/public/state.json")
    parser.add_argument(
        "--log",
        default="web/public/history.jsonl",
        help="Append a structured JSONL record (success or error) for each invocation.",
    )
    args = parser.parse_args()

    config = ConfigLoader(args.config)

    started_at = datetime.now(timezone.utc)
    record: dict = {
        "started_at": _iso_utc(started_at),
        "config": {
            "checkpoint_dir": config.checkpoint_dir,
            "webcam_url": config.webcam_url,
            "station": config.metar_station,
        },
        "model_version": git_short_sha(),
    }

    exit_code = 0
    try:
        state = predict(
            checkpoint_dir=config.checkpoint_dir,
            webcam_url=config.webcam_url,
            station=config.metar_station,
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps(state, indent=2))
        record["status"] = "ok"
        record["state"] = state
    except Exception as exc:
        record["status"] = "error"
        record["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"Inference failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        exit_code = 1
    finally:
        finished_at = datetime.now(timezone.utc)
        record["finished_at"] = _iso_utc(finished_at)
        record["duration_seconds"] = round(
            (finished_at - started_at).total_seconds(), 3
        )
        _append_log(Path(args.log), record)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

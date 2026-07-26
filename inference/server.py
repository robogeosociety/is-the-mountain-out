"""FastAPI wrapper around tools/predict_state.predict for the Cloudflare Container.

Invoked by the Cloudflare Worker on each */15 cron tick. The Worker writes the
result back to a public R2 bucket; this process is pure inference.

Checkpoint is pulled from R2 on first /predict call (cached on local FS for
subsequent calls within the container's lifetime).
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.predict_state import predict  # noqa: E402
from train.config_loader import ConfigLoader  # noqa: E402

app = FastAPI(title="Mountain Inference")
_config = ConfigLoader(os.environ.get("MOUNTAIN_CONFIG", str(ROOT / "mountain.toml")))
_storage = _config.get_storage(data_root="data")


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.post("/predict")
def run_predict() -> dict:
    try:
        state = predict(
            checkpoint_dir=_config.checkpoint_dir,
            webcam_url=_config.webcam_url,
            station=_config.metar_station,
            storage=_storage,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

    build_sha = os.environ.get("MODEL_VERSION")
    if build_sha:
        state["model_version"] = build_sha
    return state

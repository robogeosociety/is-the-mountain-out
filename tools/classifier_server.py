import os
import logging
import requests
import yaml
from pathlib import Path
from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from train.config_loader import ConfigLoader

app = FastAPI(title="Mountain Classifier API")

# Allow local dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_ROOT = Path(os.environ.get("MOUNTAIN_DATA_ROOT", "data"))
LABELS_PATH = Path(os.environ.get("MOUNTAIN_LABELS_FILE", DATA_ROOT / "labels.yaml"))

# Optional R2 storage — initialized at module level for use by endpoints
_r2_storage = None
try:
    _config = ConfigLoader(os.environ.get("MOUNTAIN_CONFIG", "mountain.toml"))
    if _config.storage_backend == "r2":
        from collect.storage import R2Storage

        _cfg = _config.storage_config
        _r2_storage = R2Storage(
            account_id=_cfg["r2_account_id"], bucket=_cfg["r2_bucket"]
        )
        logging.info(f"Classifier server: R2 storage enabled ({_cfg['r2_bucket']})")
except Exception as e:
    logging.info(f"Classifier server: R2 not configured, using local only ({e})")


class LabelBatch(BaseModel):
    labels: Dict[str, int]  # path -> label


def load_labels():
    # Pull from R2 if available (R2 is source of truth)
    if _r2_storage is not None:
        try:
            remote_text = _r2_storage.get_text("labels.yaml")
            remote_labels = yaml.safe_load(remote_text) or {}
            # Also write to local as cache
            with open(LABELS_PATH, "w") as f:
                yaml.safe_dump(remote_labels, f)
            return remote_labels
        except Exception:
            pass  # Fall through to local
    if LABELS_PATH.exists():
        with open(LABELS_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_labels(labels):
    with open(LABELS_PATH, "w") as f:
        yaml.safe_dump(labels, f)
    # Push to R2 (union merge: never delete keys)
    if _r2_storage is not None:
        try:
            # Merge with remote to avoid overwriting labels added elsewhere
            try:
                remote_text = _r2_storage.get_text("labels.yaml")
                remote_labels = yaml.safe_load(remote_text) or {}
            except Exception:
                remote_labels = {}
            remote_labels.update(labels)
            _r2_storage.put_text("labels.yaml", yaml.safe_dump(remote_labels))
        except Exception as e:
            logging.warning(f"Failed to push labels to R2: {e}")


@app.get("/api/jobs")
def get_jobs():
    """Proxy Nomad jobs for the UI."""
    try:
        nomad_url = os.environ.get("NOMAD_ADDR", "http://127.0.0.1:4646")
        response = requests.get(f"{nomad_url}/v1/jobs", timeout=2)
        response.raise_for_status()
        return response.json()
    except Exception:
        # Fallback if Nomad is unreachable
        return []


@app.get("/api/images")
def get_images(batch_size: int = 20, offset: int = 0):
    labels = load_labels()
    if _r2_storage is not None:
        all_images = sorted(k for k in _r2_storage.list_keys() if k.endswith(".jpg"))
    else:
        all_images = sorted(
            str(p.relative_to(DATA_ROOT)) for p in DATA_ROOT.rglob("*.jpg")
        )

    unlabeled = [img for img in all_images if img not in labels]

    return {
        "images": unlabeled[:batch_size],
        "total_unlabeled": len(unlabeled),
        "total_images": len(all_images),
    }


@app.get("/api/stats")
def get_stats():
    labels = load_labels()
    counts = {0: 0, 1: 0, 2: 0}
    for label in labels.values():
        counts[label] = counts.get(label, 0) + 1

    return {
        "labeled": len(labels),
        "counts": counts,
        "labels_path": str(LABELS_PATH.absolute()),
    }


@app.post("/api/label")
def post_labels(batch: LabelBatch):
    labels = load_labels()
    labels.update(batch.labels)
    save_labels(labels)
    return {"status": "success", "new_count": len(labels)}


@app.get("/api/image-url/{path:path}")
def get_image_url(path: str):
    """Return a pre-signed R2 URL for direct browser access.

    Falls back to the local /data/ static path when R2 is not configured.
    """
    if _r2_storage is not None:
        try:
            url = _r2_storage.presign(path, expires=3600)
            return {"url": url, "source": "r2"}
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to generate pre-signed URL: {e}"
            )
    return {"url": f"/data/{path}", "source": "local"}


@app.get("/api/storage-mode")
def get_storage_mode():
    """Report whether the server is using R2 or local storage."""
    return {"backend": "r2" if _r2_storage is not None else "local"}


# Serve the actual data directory for image access
# Access via /data/YYYYMMDD/...
app.mount("/data", StaticFiles(directory=DATA_ROOT), name="data")

if __name__ == "__main__":
    import uvicorn
    import socket

    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    port = int(os.environ.get("MOUNTAIN_API_PORT", 8000))
    if is_port_in_use(port):
        # Find a free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]

    # Write the actual port being used to a file so the UI or other tools can find it
    port_file = DATA_ROOT / "classifier_server.port"
    port_file.write_text(str(port))

    print(f"📡 API Server starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

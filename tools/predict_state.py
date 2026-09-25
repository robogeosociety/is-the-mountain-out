"""Single-shot inference: fetch webcam + METAR, write state.json.

This is the whole inference path since 2026-09. It runs natively on the Mac
mini once every 15 minutes, driven by `mini/tick.sh` (LaunchAgent) — no
container, no Cloudflare Worker, no network storage. It writes `state.json`,
appends `history.jsonl`, and, when the channel has something worth saying,
appends a record to `announce.jsonl` for the Discord bot to drain. Publishing
those two files to GitHub Pages is `.github/workflows/publish.yml`'s job.

The announce queue is deliberately a *file*: the tick must never block on
Discord, and a bot restart must not lose an alert.
"""

import argparse
import hashlib
import io
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from typing import Optional
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

from collect.frame import crop_burn_in  # noqa: E402
from collect.freshness import (  # noqa: E402
    FeedState,
    update_feed_state,
)
from bot.transition import (  # noqa: E402
    DEFAULT_ALERT_MIN_CONFIDENCE,
    DEFAULT_LABEL_COOLDOWN_SECONDS,
    NotifyState,
    binary_confidence,
    decide_transition,
)
from train.checkpoint_era import (  # noqa: E402
    era_matches,
    read_checkpoint_era,
)
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


def fetch_webcam_bytes(url: str, timeout: float = 20.0) -> bytes:
    """Raw JPEG bytes, kept separate from decoding so the tick can hash them.

    The hash is what tells a live camera from one that has been returning the
    same 200 for three hours (collect/freshness.py).
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def tensor_from_bytes(data: bytes, crop_bottom_px: int = 0) -> torch.Tensor:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    # Crop the station's burn-in strip BEFORE Resize, or it gets folded into
    # the rows above it rather than removed. See collect/frame.py.
    img = crop_burn_in(img, crop_bottom_px)
    return IMAGE_TRANSFORM(img).unsqueeze(0)


def fetch_webcam_tensor(
    url: str, timeout: float = 20.0, crop_bottom_px: int = 0
) -> torch.Tensor:
    return tensor_from_bytes(fetch_webcam_bytes(url, timeout), crop_bottom_px)


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


def predict(
    checkpoint_dir: str,
    webcam_url: str,
    station: str,
    storage=None,
    device: str = "mps",
    crop_bottom_px: int = 0,
    image_bytes: bytes | None = None,
) -> dict:
    # ConvNextLoRAModel downgrades to CPU on its own when MPS is unavailable, so
    # "mps" is safe to ask for on a Linux runner or an Intel Mac.
    model = ConvNextLoRAModel(
        num_classes=3, checkpoint_dir=checkpoint_dir, device=device, storage=storage
    )
    model.model_dict.eval()

    # image_bytes lets the caller fetch once and hash before deciding to run
    # the model at all — a stale feed should cost a GET, not an inference.
    if image_bytes is None:
        image_bytes = fetch_webcam_bytes(webcam_url)
    image_tensor = tensor_from_bytes(image_bytes, crop_bottom_px)
    weather_tensor, weather_readout = fetch_metar(station)

    with torch.no_grad():
        logits = model(image_tensor, weather_tensor)
        probs = F.softmax(logits, dim=1)[0].tolist()
    idx = int(max(range(3), key=lambda i: probs[i]))

    return {
        "timestamp_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "status": "ok",
        "class_index": idx,
        "class_name": CLASS_NAMES[idx],
        "is_out": idx in (1, 2),
        "confidence": {name: probs[i] for i, name in enumerate(CLASS_NAMES)},
        "weather": weather_readout,
        "webcam_url": webcam_url,
        "frame_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "model_version": git_short_sha(),
    }


def _iso_utc(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def _append_log(log_path: Path, record: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _write_state(out_path: Path, state: dict) -> None:
    """Publish-safe write: temp file + rename.

    publish.yml may read state.json at any moment, and a half-written one is a
    broken site.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state, indent=2) + "\n")
    tmp_path.replace(out_path)


def _stale_state(
    webcam_url: str,
    station: str,
    frame_sha256: str,
    now: str,
    feed: FeedState,
) -> dict:
    """What the site shows when the camera has stopped sending new frames.

    Deliberately carries no prediction. A frozen frame keeps classifying
    perfectly well — that is the danger: it would keep reporting "she's out"
    from a picture taken hours ago. `status` is the field to branch on;
    `class_name`/`is_out`/`confidence` are null so an old consumer that
    ignores `status` shows "unknown" rather than a stale answer.
    """
    weather_readout = None
    try:
        _, weather_readout = fetch_metar(station)
    except Exception:  # weather is a nicety here; the feed is the story
        pass

    return {
        "timestamp_utc": now,
        "status": "stale",
        "stale_since": feed.first_seen,
        "stale_repeat_count": feed.repeat_count,
        "class_index": None,
        "class_name": None,
        "is_out": None,
        "confidence": None,
        "weather": weather_readout,
        "webcam_url": webcam_url,
        "frame_sha256": frame_sha256,
        "model_version": git_short_sha(),
    }


def _unvalidated_state(
    webcam_url: str,
    station: str,
    frame_sha256: str,
    now: str,
    camera_era: str,
    checkpoint_era: Optional[str],
) -> dict:
    """What the site shows while no checkpoint exists for the live camera.

    Same shape as the stale document, and for the same reason: the honest
    answer is "no answer". The weights from the previous camera would load and
    produce a confident-looking number about a view they were never fit to, and
    nothing downstream could tell that apart from a real prediction — so the
    number is never computed at all.
    """
    weather_readout = None
    try:
        _, weather_readout = fetch_metar(station)
    except Exception:  # weather still renders; the model is the missing piece
        pass

    return {
        "timestamp_utc": now,
        "status": "unvalidated",
        "camera_era": camera_era,
        "checkpoint_era": checkpoint_era,
        "class_index": None,
        "class_name": None,
        "is_out": None,
        "confidence": None,
        "weather": weather_readout,
        "webcam_url": webcam_url,
        "frame_sha256": frame_sha256,
        "model_version": git_short_sha(),
    }


def _queue_label_request(
    announce_path: Path,
    notify_state_path: Path,
    state: dict,
    capture_key: Optional[str],
    label_cooldown_hours: float,
    reason: str,
) -> str:
    """Queue a bare "what is this?" post, with no prediction attached.

    The alert state machine is not consulted — there is nothing for it to
    decide without a model. But the labeling loop has to keep running, because
    reactions on these posts are the ONLY way a checkpoint for the new camera
    ever gets trained. So a frame goes out on the ordinary label cooldown,
    carrying its capture key, and 👍/⛅/👎 on it lands in labels.yaml exactly
    as before.

    Returns "label" when queued, "quiet" when the cooldown has not elapsed.
    """
    now = state["timestamp_utc"]
    try:
        stored = (
            json.loads(notify_state_path.read_text())
            if notify_state_path.exists()
            else {}
        )
    except OSError, ValueError:
        stored = {}

    last = stored.get("last_label_request")
    if last:
        try:
            elapsed = (
                datetime.fromisoformat(now.replace("Z", "+00:00"))
                - datetime.fromisoformat(last.replace("Z", "+00:00"))
            ).total_seconds()
            if elapsed < label_cooldown_hours * 3600:
                return "quiet"
        except ValueError:
            pass

    _append_log(
        announce_path,
        {
            "queued_at": now,
            "kind": "label",
            "reason": reason,
            "is_out": None,
            "class_name": None,
            "binary_confidence": None,
            "capture_key": capture_key,
            "state": state,
            "posted": False,
        },
    )

    # Only the cooldown field is touched: announced_is_out and the pending flip
    # belong to the alert machine, which has not run and must not be confused
    # by an era it knows nothing about.
    stored["last_label_request"] = now
    notify_state_path.parent.mkdir(parents=True, exist_ok=True)
    notify_state_path.write_text(json.dumps(stored, indent=2) + "\n")
    return "label"


def _queue_announcement(
    announce_path: Path,
    notify_state_path: Path,
    state: dict,
    capture_key: str | None,
    alert_min_confidence: float,
    label_cooldown_hours: float,
) -> str:
    """Run the alert/label state machine and queue anything worth posting.

    Returns the decision kind ("alert", "label" or "quiet"). Never raises into
    the tick: a broken queue must not cost us a `state.json` write, which is
    what the site actually serves.
    """
    try:
        previous = NotifyState.from_dict(
            json.loads(notify_state_path.read_text())
            if notify_state_path.exists()
            else None
        )
    except OSError, ValueError:
        previous = NotifyState()

    is_out = bool(state["is_out"])
    confidence = binary_confidence(state.get("confidence") or {}, is_out)
    now = state["timestamp_utc"]

    decision = decide_transition(
        previous,
        is_out,
        confidence,
        now,
        alert_min_confidence=alert_min_confidence,
        label_cooldown_seconds=label_cooldown_hours * 3600,
    )

    if decision.kind != "quiet":
        _append_log(
            announce_path,
            {
                "queued_at": now,
                "kind": decision.kind,
                "is_out": is_out,
                "class_name": state.get("class_name"),
                "binary_confidence": confidence,
                "previously_announced_is_out": previous.announced_is_out,
                # The frame this tick captured, as a key under the data root —
                # the bot attaches it and footers the key so a reaction becomes
                # a training label.
                "capture_key": capture_key,
                "state": state,
                "posted": False,
            },
        )

    notify_state_path.parent.mkdir(parents=True, exist_ok=True)
    notify_state_path.write_text(json.dumps(decision.next.to_dict(), indent=2) + "\n")
    return decision.kind


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
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Override [training].checkpoint_dir — on the mini this is a path on "
        "the dev disk, so inference never reaches for network storage.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Torch device. 'mps' silently falls back to CPU where MPS is absent.",
    )
    parser.add_argument(
        "--announce",
        default=None,
        help="Append alert/label-request records here for the Discord bot to "
        "drain (e.g. /Volumes/dev/mountain/live/announce.jsonl). Omit to skip.",
    )
    parser.add_argument(
        "--notify-state",
        default=None,
        help="Where the announce state machine persists (defaults to "
        "notify-state.json beside --announce).",
    )
    parser.add_argument(
        "--alert-min-confidence",
        type=float,
        default=DEFAULT_ALERT_MIN_CONFIDENCE,
        help="Binary confidence at or above which a held change is announced.",
    )
    parser.add_argument(
        "--label-cooldown-hours",
        type=float,
        default=DEFAULT_LABEL_COOLDOWN_SECONDS / 3600,
        help="Minimum gap between label requests.",
    )
    parser.add_argument(
        "--camera-era",
        default=None,
        help="Override [webcam] era. A checkpoint that does not declare this "
        "era is refused and the site publishes status=unvalidated.",
    )
    parser.add_argument(
        "--feed-state",
        default=None,
        help="Where the frame-freshness bookkeeping persists (defaults to "
        "feed-state.json beside --out).",
    )
    parser.add_argument(
        "--stale-after-repeats",
        type=int,
        default=None,
        help="Consecutive byte-identical frames before the feed is declared "
        "stale and no prediction is written (default: [webcam] "
        "stale_after_repeats in the config).",
    )
    parser.add_argument(
        "--capture-key",
        default=None,
        help="Key of the frame this tick captured, relative to the data root; "
        "carried into the announce record so a reaction becomes a label.",
    )
    args = parser.parse_args()

    config = ConfigLoader(args.config)
    checkpoint_dir = args.checkpoint_dir or config.checkpoint_dir

    started_at = datetime.now(timezone.utc)
    record: dict = {
        "started_at": _iso_utc(started_at),
        "config": {
            "checkpoint_dir": checkpoint_dir,
            "webcam_url": config.webcam_url,
            "station": config.metar_station,
        },
        "model_version": git_short_sha(),
    }

    exit_code = 0
    out_path = Path(args.out)
    try:
        # Fetch first and hash the bytes. A dead feed should cost one GET, not
        # a model load — and it must not be answered with a prediction.
        image_bytes = fetch_webcam_bytes(config.webcam_url)
        frame_sha256 = hashlib.sha256(image_bytes).hexdigest()
        now = _iso_utc(datetime.now(timezone.utc))

        feed_state_path = (
            Path(args.feed_state)
            if args.feed_state
            else out_path.parent / "feed-state.json"
        )
        try:
            previous_feed = FeedState.from_dict(
                json.loads(feed_state_path.read_text())
                if feed_state_path.exists()
                else None
            )
        except OSError, ValueError:
            previous_feed = FeedState()

        feed, is_stale = update_feed_state(
            previous_feed,
            frame_sha256,
            now,
            stale_after_repeats=args.stale_after_repeats
            if args.stale_after_repeats is not None
            else config.webcam_stale_after_repeats,
        )
        feed_state_path.parent.mkdir(parents=True, exist_ok=True)
        feed_state_path.write_text(json.dumps(feed.to_dict(), indent=2) + "\n")
        record["feed"] = {**feed.to_dict(), "stale": is_stale}

        camera_era = args.camera_era or config.camera_era
        checkpoint_era = read_checkpoint_era(
            checkpoint_dir, fallback=config.checkpoint_era_fallback
        )
        # An unset camera era opts out of the gate entirely; otherwise the
        # checkpoint has to name the same camera it will be asked about.
        era_ok = not camera_era or era_matches(camera_era, checkpoint_era)
        record["era"] = {
            "camera": camera_era,
            "checkpoint": checkpoint_era,
            "match": era_ok,
        }

        if not era_ok and not is_stale:
            # No model for this camera. Do not load the old one: it would
            # produce a confident number about a view it never saw, and
            # nothing downstream could tell that from a real prediction.
            state = _unvalidated_state(
                config.webcam_url,
                config.metar_station,
                frame_sha256,
                now,
                camera_era,
                checkpoint_era,
            )
            _write_state(out_path, state)
            print(json.dumps(state, indent=2))
            print(
                f"No checkpoint for camera era {camera_era!r} "
                f"(checkpoint is {checkpoint_era!r}). No prediction written.",
                file=sys.stderr,
            )
            record["status"] = "unvalidated"
            record["state"] = state

            # The labeling loop keeps running: reactions on these posts are the
            # only path to a checkpoint that WOULD be valid here.
            if args.announce:
                announce_path = Path(args.announce)
                notify_state_path = (
                    Path(args.notify_state)
                    if args.notify_state
                    else announce_path.parent / "notify-state.json"
                )
                try:
                    record["announced"] = _queue_label_request(
                        announce_path,
                        notify_state_path,
                        state,
                        args.capture_key,
                        args.label_cooldown_hours,
                        reason="unvalidated",
                    )
                except Exception as exc:
                    record["announced"] = "error"
                    print(
                        f"Label queue failed: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
            return exit_code

        if is_stale:
            # Same bytes N ticks running: the camera is frozen or the CDN is
            # serving a cached corpse. Say so; do not guess.
            state = _stale_state(
                config.webcam_url, config.metar_station, frame_sha256, now, feed
            )
            _write_state(out_path, state)
            print(json.dumps(state, indent=2))
            print(
                f"Feed stale: identical frame {feed.repeat_count}x since "
                f"{feed.first_seen} ({frame_sha256[:12]}). No prediction written.",
                file=sys.stderr,
            )
            record["status"] = "stale"
            record["state"] = state
            # No announcement: a stale feed is a pipeline fault, not news about
            # the mountain, and the alert state machine must not consume it.
            return exit_code

        state = predict(
            checkpoint_dir=checkpoint_dir,
            webcam_url=config.webcam_url,
            station=config.metar_station,
            device=args.device,
            crop_bottom_px=config.webcam_crop_bottom_px,
            image_bytes=image_bytes,
        )
        _write_state(out_path, state)
        print(json.dumps(state, indent=2))
        record["status"] = "ok"
        record["state"] = state

        if args.announce:
            announce_path = Path(args.announce)
            notify_state_path = (
                Path(args.notify_state)
                if args.notify_state
                else announce_path.parent / "notify-state.json"
            )
            try:
                record["announced"] = _queue_announcement(
                    announce_path,
                    notify_state_path,
                    state,
                    args.capture_key,
                    args.alert_min_confidence,
                    args.label_cooldown_hours,
                )
            except Exception as exc:  # the site matters more than the channel
                record["announced"] = "error"
                print(
                    f"Announce queue failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
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

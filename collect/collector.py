import click
import time
import json
import requests
import threading
from pathlib import Path
from datetime import datetime, UTC
import cv2
from typing import Optional, Dict, List
import uuid
import yaml
import logging
import rumps

# Assuming 'train' and 'collect' are in the python path.
from train.config_loader import ConfigLoader
from collect.tray import MountainTray
from collect.state import make_state, write_state, read_state, write_plan, read_plan
from collect.sync import sync

# --- Globals ---
LOG_FILE = "data/collection.log"

# --- Core Business Logic ---


class WebcamStream:
    def __init__(self, url: str):
        self.url = url
        self.cap = cv2.VideoCapture(url)

    def capture_raw(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def release(self):
        self.cap.release()


class WeatherFetcher:
    def __init__(self, station_id: str = "KSEA"):
        self.station_id = station_id
        self.base_url = f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{station_id}.TXT"

    def fetch_latest_metar(self) -> Optional[str]:
        try:
            response = requests.get(self.base_url, timeout=10)
            response.raise_for_status()
            lines = response.text.strip().split("\\n")
            return lines[1] if len(lines) >= 2 else lines[0]
        except requests.RequestException as e:
            logging.warning(f"Error fetching METAR: {e}")
        return None


def _derive_initial_last_capture_at(
    plan_timestamps: List[str],
    data_root: str,
    now_utc,
    session_id: str,
) -> Optional[str]:
    """Return best known last_capture_at on startup.

    Priority: most recent past plan timestamp → previous state file value
    (only if it's actually in the past).
    """
    from datetime import datetime

    past = [t for t in plan_timestamps if datetime.fromisoformat(t) <= now_utc]
    if past:
        return past[-1]
    prev_state = read_state(data_root, session_id)
    if prev_state and prev_state.last_capture_at:
        try:
            prev_ts = datetime.fromisoformat(prev_state.last_capture_at)
            if prev_ts <= now_utc:
                return prev_state.last_capture_at
        except ValueError:
            pass
    return None


def log_event(event: str, status: str, metadata: Optional[Dict] = None):
    log_path = Path(LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": event,
        "status": status,
        "metadata": metadata or {},
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def perform_capture(
    config_loader: ConfigLoader,
    weather_fetcher: WeatherFetcher,
    data_root: str,
    session_uuid: Optional[str] = None,
    step_info: Optional[Dict] = None,
    remote_storage=None,
) -> Optional[Path]:
    now_utc = datetime.now(UTC)
    date_str = now_utc.strftime("%Y%m%d")
    time_str = now_utc.strftime("%H%M%S_%f_UTC")

    capture_dir = Path(data_root) / date_str / time_str
    image_dir = capture_dir / "images"
    metar_dir = capture_dir / "metar"

    image_dir.mkdir(parents=True, exist_ok=True)
    metar_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Starting collection into {capture_dir}...")

    metar_text = weather_fetcher.fetch_latest_metar()
    if metar_text:
        metar_path = metar_dir / "metar.txt"
        metar_path.write_text(metar_text)
        logging.info("METAR data saved.")
    else:
        metar_path = None
        logging.warning("METAR capture failed.")

    source = config_loader.webcam_url
    if not source:
        logging.error("No webcam source configured.")
        return False

    stream = WebcamStream(source)
    try:
        frame = stream.capture_raw()
        if frame is not None:
            safe_name = (
                str(source).replace("/", "_").replace(":", "_").replace(".", "_")
            )
            filename = f"{time_str}_{safe_name}.jpg"
            image_path = image_dir / filename
            cv2.imwrite(str(image_path), frame)
            logging.info(f"Source {source}: Saved as {filename}")

            # Upload to R2 if a remote storage backend is provided
            if remote_storage is not None:
                _upload_to_remote(
                    remote_storage, data_root, image_path, frame, metar_path, metar_text
                )

            return image_path
        else:
            logging.warning(f"Source {source}: Capture failed.")
            return None
    finally:
        stream.release()


def _upload_to_remote(
    storage,
    data_root: str,
    image_path: Path,
    frame,
    metar_path: Optional[Path],
    metar_text: Optional[str],
) -> None:
    """Best-effort upload of a capture to the remote storage backend."""
    try:
        root = Path(data_root)
        # Encode the frame as JPEG bytes
        _, buf = cv2.imencode(".jpg", frame)
        storage.put(str(image_path.relative_to(root)), buf.tobytes())
        if metar_text and metar_path:
            storage.put_text(str(metar_path.relative_to(root)), metar_text)
        logging.info("R2 upload complete.")
    except Exception as e:
        logging.warning(f"R2 upload failed (non-fatal): {e}")


# --- CLI using Click ---


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx, **kwargs):
    """A tool to collect images of Mount Rainier. Defaults to running with a tray icon."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(tray)


def run_tray_loop(
    config_path: str,
    data_root: str,
    is_once: bool = False,
    session_id: Optional[str] = None,
):
    """Internal implementation for running the tray icon loop."""
    from datetime import datetime, timezone

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not session_id:
        session_id = str(uuid.uuid4())[:8]
    config_loader = ConfigLoader(config_path)
    weather_fetcher = WeatherFetcher(config_loader.metar_station)
    fallback_interval = config_loader.collection_seconds

    # Optional remote storage for R2 uploads
    remote_storage = None
    if config_loader.storage_backend == "r2":
        try:
            from collect.storage import R2Storage

            cfg = config_loader.storage_config
            remote_storage = R2Storage(
                account_id=cfg["r2_account_id"], bucket=cfg["r2_bucket"]
            )
            logging.info(f"R2 upload enabled: {cfg['r2_bucket']}")
        except Exception as e:
            logging.warning(f"R2 storage init failed, continuing local-only: {e}")

    # Load plan if one exists
    all_plan_timestamps = read_plan(data_root) or []
    now_utc = datetime.now(timezone.utc)
    plan_timestamps = [
        t for t in all_plan_timestamps if datetime.fromisoformat(t) > now_utc
    ]
    if all_plan_timestamps:
        logging.info(f"Loaded plan: {len(plan_timestamps)} captures remaining.")
    else:
        logging.info(
            f"No plan file found. Using fixed interval ({fallback_interval}s)."
        )

    plan_last_capture_at = _derive_initial_last_capture_at(
        all_plan_timestamps, data_root, now_utc, session_id
    )
    plan_final_capture_at = all_plan_timestamps[-1] if all_plan_timestamps else None

    plan_total = len(plan_timestamps) if plan_timestamps else 0

    # Seed capture count from previous state so restarts don't reset to zero.
    _prev = read_state(data_root, session_id)
    capture_count = _prev.capture_count if _prev else 0
    # plan_index tracks position within plan_timestamps (future-only list).
    # Always start at 0 since plan_timestamps is already filtered to future.
    plan_index = 0

    # Load carry-over counts from prior sessions toward the same goal.
    prior_capture_count = 0
    prior_plan_total = 0
    _prior_path = Path(data_root) / "prior_sessions.json"
    try:
        _prior = json.loads(_prior_path.read_text())
        prior_capture_count = _prior.get("capture_count", 0)
        prior_plan_total = _prior.get("plan_total", 0)
        logging.info(
            f"Prior sessions: {prior_capture_count}/{prior_plan_total} captures carried over."
        )
    except Exception:
        pass

    stop_event = threading.Event()
    last_capture_at = plan_last_capture_at  # tracked across all loop iterations
    session_labels_file = Path(data_root) / f"labels.{session_id}.yaml"
    trigger_file = Path(data_root) / f"trigger_{session_id}"

    def _append_session_label(image_path: Path, is_adhoc: bool = False) -> None:
        rel = str(image_path.relative_to(Path(data_root)))
        entry = {rel: {"type": "manual" if is_adhoc else "scheduled"}}
        with open(session_labels_file, "a") as f:
            yaml.dump(entry, f, default_flow_style=False)

    def _next_capture_at() -> Optional[str]:
        if plan_timestamps and plan_index < len(plan_timestamps):
            return plan_timestamps[plan_index]
        return None

    def _sleep_until_next() -> bool:
        """Sleep until the next scheduled capture time, or fallback interval.
        Returns True if a trigger file was detected.
        """
        next_iso = _next_capture_at()
        if next_iso:
            target = datetime.fromisoformat(next_iso)
            wait = max(0, (target - datetime.now(timezone.utc)).total_seconds())
        else:
            wait = fallback_interval

        logging.info(f"Next capture in {int(wait)}s.")
        end = time.monotonic() + wait
        while time.monotonic() < end:
            if stop_event.is_set():
                return False
            # Check for ad-hoc trigger file
            if trigger_file.exists():
                logging.info("Ad-hoc trigger detected!")
                try:
                    trigger_file.unlink()
                except Exception:
                    pass
                return True
            time.sleep(min(1, end - time.monotonic()))
        return False

    def capture_loop():
        nonlocal \
            capture_count, \
            plan_index, \
            last_capture_at, \
            plan_total, \
            plan_final_capture_at

        while not stop_event.is_set():
            # Scheduled capture start
            write_state(
                data_root,
                make_state(
                    session_id=session_id,
                    status="Capturing...",
                    capture_count=capture_count,
                    plan_total=plan_total,
                    interval_seconds=fallback_interval,
                    last_capture_at=last_capture_at,
                    next_capture_at=_next_capture_at(),
                    session_labels_file=str(session_labels_file),
                    final_capture_at=plan_final_capture_at,
                    prior_capture_count=prior_capture_count,
                    prior_plan_total=prior_plan_total,
                ),
            )

            image_path = perform_capture(
                config_loader,
                weather_fetcher,
                data_root,
                session_uuid=session_id,
                remote_storage=remote_storage,
            )

            if image_path:
                capture_count += 1
                last_capture_at = datetime.now(timezone.utc).isoformat()
                _append_session_label(image_path, is_adhoc=False)

            if plan_timestamps:
                plan_index += 1

            write_state(
                data_root,
                make_state(
                    session_id=session_id,
                    status="Idle" if image_path else "Error",
                    capture_count=capture_count,
                    plan_total=plan_total,
                    interval_seconds=fallback_interval,
                    last_capture_at=last_capture_at,
                    next_capture_at=_next_capture_at(),
                    session_labels_file=str(session_labels_file),
                    final_capture_at=plan_final_capture_at,
                    prior_capture_count=prior_capture_count,
                    prior_plan_total=prior_plan_total,
                ),
            )

            if is_once:
                logging.info("Single capture complete.")
                stop_event.set()
                rumps.quit_application()
                return

            if plan_timestamps and plan_index >= len(plan_timestamps):
                logging.info("Plan complete. Regenerating for next 30 days...")
                try:
                    import sys as _sys
                    from datetime import timedelta as _td

                    _sys.path.insert(0, str(Path(config_path).resolve().parent))
                    from tools.plan import CapturePlan

                    planner = CapturePlan(47.6533, -122.3091)
                    _now = datetime.now(timezone.utc).replace(
                        minute=0, second=0, microsecond=0
                    )
                    intervals = planner.generate(_now, days=30)
                    new_ts = []
                    current = _now
                    for step in intervals:
                        if step == "stop":
                            break
                        current = current + _td(seconds=int(step[:-1]))
                        new_ts.append(current.isoformat())
                    write_plan(data_root, new_ts)
                    plan_timestamps[:] = [
                        t
                        for t in new_ts
                        if datetime.fromisoformat(t) > datetime.now(timezone.utc)
                    ]
                    plan_index = 0
                    plan_total = len(plan_timestamps)
                    plan_final_capture_at = new_ts[-1] if new_ts else None
                    logging.info(
                        f"New plan: {len(plan_timestamps)} captures over 30 days."
                    )
                except Exception:
                    logging.exception(
                        "Failed to regenerate plan. Falling back to fixed interval."
                    )
                    plan_timestamps.clear()
                    plan_index = 0
                    plan_total = 0

            # Wait for next or trigger
            while not stop_event.is_set():
                is_adhoc = _sleep_until_next()
                if not is_adhoc:
                    # Normal scheduled wakeup
                    break

                # Ad-hoc capture wakeup
                write_state(
                    data_root,
                    make_state(
                        session_id=session_id,
                        status="Capturing (Ad-hoc)...",
                        capture_count=capture_count,
                        plan_total=plan_total,
                        interval_seconds=fallback_interval,
                        last_capture_at=last_capture_at,
                        next_capture_at=_next_capture_at(),
                        session_labels_file=str(session_labels_file),
                        final_capture_at=plan_final_capture_at,
                    ),
                )

                image_path = perform_capture(
                    config_loader,
                    weather_fetcher,
                    data_root,
                    session_uuid=session_id,
                    remote_storage=remote_storage,
                )
                if image_path:
                    capture_count += 1
                    last_capture_at = datetime.now(timezone.utc).isoformat()
                    _append_session_label(image_path, is_adhoc=True)

                write_state(
                    data_root,
                    make_state(
                        session_id=session_id,
                        status="Idle",
                        capture_count=capture_count,
                        plan_total=plan_total,
                        interval_seconds=fallback_interval,
                        last_capture_at=last_capture_at,
                        next_capture_at=_next_capture_at(),
                        session_labels_file=str(session_labels_file),
                        final_capture_at=plan_final_capture_at,
                    ),
                )

    # Write initial state before starting tray so first _refresh() has data.
    # last_capture_at is derived from the plan (most recent past timestamp).
    write_state(
        data_root,
        make_state(
            session_id=session_id,
            status="Starting...",
            capture_count=capture_count,
            plan_total=plan_total,
            interval_seconds=fallback_interval,
            last_capture_at=last_capture_at,
            next_capture_at=_next_capture_at(),
            session_labels_file=str(session_labels_file),
            final_capture_at=plan_final_capture_at,
        ),
    )

    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()

    if rumps:
        tray_manager = MountainTray(data_root=data_root, session_id=session_id)
        tray_manager.run()
    else:
        logging.info("Headless mode: Waiting for loop to complete...")
        while not stop_event.is_set():
            time.sleep(1)


@cli.command()
@click.option("--data-root", default="data", help="Root directory for data storage.")
@click.option("--days", default=30, help="Number of days to plan ahead.")
@click.option("--lat", default=47.6533, help="Latitude of camera location.")
@click.option("--lon", default=-122.3091, help="Longitude of camera location.")
def schedule(data_root: str, days: int, lat: float, lon: float):
    """Generate a solar-aligned capture plan and save it to capture_plan.json."""
    import sys as _sys
    from datetime import datetime, timezone, timedelta as _timedelta

    _sys.path.insert(0, str(Path(__file__).parent.parent))
    from tools.plan import CapturePlan

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    planner = CapturePlan(lat, lon)
    intervals = planner.generate(now, days=days)

    # Convert wait-time intervals to absolute UTC timestamps
    timestamps = []
    current = now
    for step in intervals:
        if step == "stop":
            break
        wait = int(step[:-1])
        current = current + _timedelta(seconds=wait)
        timestamps.append(current.isoformat())

    path = write_plan(data_root, timestamps)
    click.echo(f"Plan saved: {path}")
    click.echo(f"  {len(timestamps)} captures over {days} days")
    click.echo(f"  First: {timestamps[0]}")
    click.echo(f"  Last:  {timestamps[-1]}")


@cli.command()
@click.option("--config", default="mountain.toml", help="Path to config file.")
@click.option("--data-root", default="data", help="Root directory for data storage.")
@click.option("--session-id", default=None, help="Unique ID for this session.")
def tray(config: str, data_root: str, session_id: str):
    """Runs continuous collection with a system tray icon."""
    run_tray_loop(config, data_root, is_once=False, session_id=session_id)


@cli.command()
@click.option("--config", default="mountain.toml", help="Path to config file.")
@click.option("--data-root", default="data", help="Root directory for data storage.")
@click.option("--session-id", default=None, help="Unique ID for this session.")
def once(config: str, data_root: str, session_id: str):
    """Performs a single capture, showing a tray icon briefly."""
    run_tray_loop(config, data_root, is_once=True, session_id=session_id)


@cli.command()
@click.option("--config", default="mountain.toml", help="Path to config file.")
@click.option("--data-root", default="data", help="Root directory for data storage.")
def live(config: str, data_root: str):
    """Runs continuous collection in the foreground (no tray icon)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    session_id = str(uuid.uuid4())[:8]
    config_loader = ConfigLoader(config)
    weather_fetcher = WeatherFetcher(config_loader.metar_station)
    interval = config_loader.collection_seconds

    remote_storage = None
    if config_loader.storage_backend == "r2":
        try:
            from collect.storage import R2Storage

            cfg = config_loader.storage_config
            remote_storage = R2Storage(
                account_id=cfg["r2_account_id"], bucket=cfg["r2_bucket"]
            )
            logging.info(f"R2 upload enabled: {cfg['r2_bucket']}")
        except Exception as e:
            logging.warning(f"R2 storage init failed, continuing local-only: {e}")

    logging.info(
        f"Starting live collection loop (interval: {interval}s). Press Ctrl+C to stop."
    )
    try:
        while True:
            perform_capture(
                config_loader,
                weather_fetcher,
                data_root,
                session_uuid=session_id,
                remote_storage=remote_storage,
            )
            time.sleep(interval)
    except KeyboardInterrupt:
        logging.info("Stopping live collection.")


cli.add_command(sync)

if __name__ == "__main__":
    cli()

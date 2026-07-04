"""CLI subcommands for syncing data between local filesystem and R2."""

import click
import yaml
import logging
from pathlib import Path

from train.config_loader import ConfigLoader
from collect.storage import LocalStorage, R2Storage

logger = logging.getLogger(__name__)


def _get_r2(config_path: str) -> R2Storage:
    config = ConfigLoader(config_path)
    if config.storage_backend != "r2":
        raise click.ClickException(
            "R2 storage is not configured. Add [storage] section to mountain.toml."
        )
    cfg = config.storage_config
    return R2Storage(account_id=cfg["r2_account_id"], bucket=cfg["r2_bucket"])


@click.group()
def sync():
    """Sync data between local filesystem and Cloudflare R2."""
    pass


@sync.command()
@click.option("--config", default="mountain.toml", help="Path to config file.")
@click.option("--data-root", default="data", help="Local data root.")
def push(config: str, data_root: str):
    """Upload all local captures to R2 (skips files that already exist in R2)."""
    logging.basicConfig(level=logging.INFO)
    r2 = _get_r2(config)
    local = LocalStorage(data_root)

    all_keys = local.list_keys()
    # Filter to capture data only (images + metar), skip state/config files
    capture_keys = [
        k for k in all_keys if "/" in k and (k.endswith(".jpg") or k.endswith(".txt"))
    ]

    uploaded = 0
    skipped = 0
    for key in capture_keys:
        if r2.exists(key):
            skipped += 1
            continue
        try:
            data = local.get(key)
            r2.put(key, data)
            uploaded += 1
        except Exception as e:
            logger.warning(f"Failed to upload {key}: {e}")

    click.echo(
        f"Push complete: {uploaded} uploaded, {skipped} skipped (already in R2)."
    )


@sync.command()
@click.option("--config", default="mountain.toml", help="Path to config file.")
@click.option("--data-root", default="data", help="Local data root.")
def pull(config: str, data_root: str):
    """Download R2 captures to local filesystem (skips files that already exist locally)."""
    logging.basicConfig(level=logging.INFO)
    r2 = _get_r2(config)
    local = LocalStorage(data_root)

    all_keys = r2.list_keys()
    # Filter to capture data only
    capture_keys = [
        k for k in all_keys if "/" in k and (k.endswith(".jpg") or k.endswith(".txt"))
    ]

    downloaded = 0
    skipped = 0
    for key in capture_keys:
        if local.exists(key):
            skipped += 1
            continue
        try:
            data = r2.get(key)
            local.put(key, data)
            downloaded += 1
        except Exception as e:
            logger.warning(f"Failed to download {key}: {e}")

    click.echo(
        f"Pull complete: {downloaded} downloaded, {skipped} skipped (already local)."
    )


@sync.group()
def labels():
    """Sync labels.yaml between local and R2."""
    pass


@labels.command("push")
@click.option("--config", default="mountain.toml", help="Path to config file.")
@click.option("--data-root", default="data", help="Local data root.")
def labels_push(config: str, data_root: str):
    """Union-merge local labels.yaml into R2 (never deletes remote labels)."""
    logging.basicConfig(level=logging.INFO)
    r2 = _get_r2(config)

    labels_path = Path(data_root) / "labels.yaml"
    if not labels_path.exists():
        raise click.ClickException(f"Local labels file not found: {labels_path}")

    with open(labels_path, "r") as f:
        local_labels = yaml.safe_load(f) or {}

    # Merge with remote (union — local wins on conflict, remote keys preserved)
    try:
        remote_text = r2.get_text("labels.yaml")
        remote_labels = yaml.safe_load(remote_text) or {}
    except Exception:
        remote_labels = {}

    merged = {**remote_labels, **local_labels}
    r2.put_text("labels.yaml", yaml.safe_dump(merged))

    added = len(merged) - len(remote_labels)
    click.echo(
        f"Labels pushed: {len(merged)} total ({added} new, {len(local_labels)} from local)."
    )


@labels.command("pull")
@click.option("--config", default="mountain.toml", help="Path to config file.")
@click.option("--data-root", default="data", help="Local data root.")
def labels_pull(config: str, data_root: str):
    """Overwrite local labels.yaml from R2 (R2 is source of truth)."""
    logging.basicConfig(level=logging.INFO)
    r2 = _get_r2(config)

    try:
        remote_text = r2.get_text("labels.yaml")
    except Exception as e:
        raise click.ClickException(f"Failed to read labels from R2: {e}")

    labels_path = Path(data_root) / "labels.yaml"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(remote_text)

    remote_labels = yaml.safe_load(remote_text) or {}
    click.echo(f"Labels pulled: {len(remote_labels)} labels written to {labels_path}.")

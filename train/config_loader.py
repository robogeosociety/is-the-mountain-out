import os

import tomli
from typing import Dict, Any, Optional

#: [paths] — the LOCAL training tree, used on whichever machine runs
#: `just train`. Air-only since 2026-09-27; the mini is 8 GB and does not
#: train. Defaults mirror the mini's layout so a pulled tree is drop-in.
DEFAULT_PATHS = {
    "captures_dir": "~/mountain/data",
    "labels_dir": "~/mountain/data",
    "checkpoint_dir": "~/mountain/checkpoints",
}

#: [mini] — the always-on host: where labels come from and where a trained
#: checkpoint goes back to.
DEFAULT_MINI_HOST = "tommydoerr@tommys-mac-mini.local"
DEFAULT_MINI_ROOT = "/Users/tommydoerr/mountain"
DEFAULT_TICK_LABEL = "com.robogeosociety.mountain-tick"


class ConfigLoader:
    def __init__(self, config_path: str = "mountain.toml"):
        self.config_path = config_path
        self.data = self._load_toml_config(self.config_path)

        self._validate_config()

    def _load_toml_config(self, path: str) -> Dict[str, Any]:
        with open(path, "rb") as f:
            return tomli.load(f)

    def _validate_config(self):
        # Basic config validation
        required_sections = ["mountain", "webcam", "weather", "training", "collection"]
        for section in required_sections:
            if section not in self.data:
                raise ValueError(f"Missing required config section: {section}")

    @property
    def webcam_url(self) -> str:
        return self.data["webcam"].get("url", "")

    @property
    def camera_era(self) -> Optional[str]:
        """Era string the live camera declares, or None to disable the gate."""
        era = self.data["webcam"].get("era")
        return era.strip() if isinstance(era, str) and era.strip() else None

    @property
    def checkpoint_era_fallback(self) -> Optional[str]:
        """Era to assume for a checkpoint that carries no era.json."""
        era = self.data["training"].get("checkpoint_era")
        return era.strip() if isinstance(era, str) and era.strip() else None

    @property
    def webcam_crop_bottom_px(self) -> int:
        """Burn-in strip to drop before the model sees a frame.

        Defaults to 0 — a camera without a burn-in strip must not silently
        lose rows because another camera had one.
        """
        return int(self.data["webcam"].get("crop_bottom_px", 0) or 0)

    @property
    def webcam_stale_after_repeats(self) -> int:
        """Consecutive byte-identical fetches before the feed is called dead."""
        from collect.freshness import DEFAULT_STALE_AFTER_REPEATS

        return int(
            self.data["webcam"].get("stale_after_repeats", DEFAULT_STALE_AFTER_REPEATS)
        )

    @property
    def schedule_seconds(self) -> int:
        return self.data["training"].get("schedule_seconds", 1800)

    @property
    def collection_seconds(self) -> int:
        return self.data["collection"].get("collection_seconds", 600)

    @property
    def collection_schedule(self) -> Optional[Dict[str, int]]:
        return self.data["collection"].get("schedule")

    @property
    def capture_interval_seconds(self) -> int:
        return self.data["training"].get("capture_interval_seconds", 300)

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.data["training"].get("gradient_accumulation_steps", 4)

    @property
    def lora_settings(self) -> Dict[str, Any]:
        return self.data["training"].get("lora", {})

    @property
    def checkpoint_dir(self) -> str:
        return self.data["training"].get("checkpoint_dir", "train/checkpoints")

    @property
    def metar_station(self) -> str:
        return self.data["weather"].get("station_id", "KSEA")

    @property
    def mountain_data(self) -> Dict[str, Any]:
        return self.data.get("mountain", {})

    # -- Where the two machines keep things (Air trains, mini serves) --

    @property
    def paths(self) -> Dict[str, str]:
        """[paths] — the local training tree, with `~` and $VARS expanded.

        Expansion happens here because these are read on the machine that
        owns them. The [mini] equivalents below are deliberately NOT expanded:
        they name directories on the other end of an rsync, and resolving
        `~` against the Air's home would silently point at the wrong place.
        """
        section = self.data.get("paths", {})
        return {
            key: os.path.expanduser(
                os.path.expandvars(str(section.get(key, default)))
            ).rstrip("/")
            for key, default in DEFAULT_PATHS.items()
        }

    @property
    def mini(self) -> Dict[str, str]:
        """[mini] — the always-on host, and its copy of the same tree.

        `captures_dir`/`labels_dir`/`checkpoint_dir` derive from `root` unless
        they are set explicitly, so the common case is two lines of config.
        """
        section = self.data.get("mini", {})
        root = str(section.get("root", DEFAULT_MINI_ROOT)).rstrip("/")
        return {
            "host": str(section.get("host", DEFAULT_MINI_HOST)),
            "root": root,
            "captures_dir": str(section.get("captures_dir", f"{root}/data")).rstrip(
                "/"
            ),
            "labels_dir": str(section.get("labels_dir", f"{root}/data")).rstrip("/"),
            "checkpoint_dir": str(
                section.get("checkpoint_dir", f"{root}/checkpoints")
            ).rstrip("/"),
            "tick_label": str(section.get("tick_label", DEFAULT_TICK_LABEL)),
        }

    # -- Optional storage backend (R2) --

    @property
    def storage_backend(self) -> str:
        """Return 'local' or 'r2'. Defaults to 'local' when [storage] is absent."""
        return self.data.get("storage", {}).get("backend", "local")

    @property
    def storage_config(self) -> Dict[str, Any]:
        """Return the raw [storage] section, or empty dict."""
        return self.data.get("storage", {})

    def get_storage(self, data_root: str = "data"):
        """Factory: return the appropriate StorageBackend based on config.

        Returns LocalStorage when backend='local' or [storage] is absent.
        Returns R2Storage (or CachedR2Storage) when backend='r2'.
        """
        from collect.storage import LocalStorage, R2Storage, CachedR2Storage

        if self.storage_backend != "r2":
            return LocalStorage(data_root)

        cfg = self.storage_config
        r2 = R2Storage(
            account_id=cfg["r2_account_id"],
            bucket=cfg["r2_bucket"],
        )
        cache_dir = cfg.get("cache_dir", ".r2cache")
        return CachedR2Storage(r2, cache_dir=cache_dir)

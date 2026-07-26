import tomli
from typing import Dict, Any, Optional


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

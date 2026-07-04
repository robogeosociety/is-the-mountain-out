"""Tests for ConfigLoader storage properties and get_storage() factory."""

import tomli_w
from unittest.mock import patch, MagicMock
from train.config_loader import ConfigLoader


def _base_config():
    """Minimal valid config without [storage]."""
    return {
        "mountain": {"name": "Mount Rainier", "height": 14411},
        "webcam": {"url": "http://cam.jpg", "name": "Cam"},
        "weather": {"station_id": "KSEA"},
        "training": {
            "schedule_seconds": 1800,
            "capture_interval_seconds": 300,
            "gradient_accumulation_steps": 4,
            "checkpoint_dir": "checkpoints",
            "lora": {"rank": 4, "alpha": 8, "target_modules": ["fc1"]},
        },
        "collection": {"collection_seconds": 600},
    }


def _write_config(tmp_path, data):
    config_file = tmp_path / "mountain.toml"
    config_file.write_bytes(tomli_w.dumps(data).encode())
    return str(config_file)


class TestStorageConfigDefaults:
    def test_no_storage_section(self, tmp_path):
        loader = ConfigLoader(_write_config(tmp_path, _base_config()))
        assert loader.storage_backend == "local"
        assert loader.storage_config == {}

    def test_storage_local_explicit(self, tmp_path):
        cfg = _base_config()
        cfg["storage"] = {"backend": "local"}
        loader = ConfigLoader(_write_config(tmp_path, cfg))
        assert loader.storage_backend == "local"

    def test_storage_r2(self, tmp_path):
        cfg = _base_config()
        cfg["storage"] = {
            "backend": "r2",
            "r2_account_id": "abc123",
            "r2_bucket": "my-bucket",
            "cache_dir": ".cache",
        }
        loader = ConfigLoader(_write_config(tmp_path, cfg))
        assert loader.storage_backend == "r2"
        assert loader.storage_config["r2_bucket"] == "my-bucket"
        assert loader.storage_config["cache_dir"] == ".cache"


class TestGetStorageFactory:
    def test_local_default(self, tmp_path):
        from collect.storage import LocalStorage

        loader = ConfigLoader(_write_config(tmp_path, _base_config()))
        storage = loader.get_storage(str(tmp_path / "data"))
        assert isinstance(storage, LocalStorage)

    @patch.dict(
        "sys.modules",
        {"boto3": MagicMock(), "botocore": MagicMock(), "botocore.config": MagicMock()},
    )
    def test_r2_returns_cached(self, tmp_path, monkeypatch):
        monkeypatch.setenv("R2_ACCESS_KEY_ID", "k")
        monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "s")

        # Reload storage module so it picks up the mocked boto3
        import importlib
        import collect.storage

        importlib.reload(collect.storage)
        from collect.storage import CachedR2Storage

        cfg = _base_config()
        cfg["storage"] = {
            "backend": "r2",
            "r2_account_id": "abc123",
            "r2_bucket": "test-bucket",
        }
        loader = ConfigLoader(_write_config(tmp_path, cfg))
        storage = loader.get_storage(str(tmp_path / "data"))
        assert isinstance(storage, CachedR2Storage)
        assert storage.r2.bucket == "test-bucket"

    def test_validation_still_passes_without_storage(self, tmp_path):
        """[storage] is optional — validation should not require it."""
        loader = ConfigLoader(_write_config(tmp_path, _base_config()))
        # Should not raise
        assert loader.webcam_url == "http://cam.jpg"

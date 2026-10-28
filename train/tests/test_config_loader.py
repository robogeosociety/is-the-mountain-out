import tomli_w
from train.config_loader import ConfigLoader


def test_config_loader(tmp_path):
    # Consolidated mountain.toml
    config_file = tmp_path / "mountain.toml"
    config_data = {
        "mountain": {"name": "Mount Rainier", "height": 14411},
        "webcam": {"url": "http://rainier-webcam.jpg", "name": "Rainier View"},
        "weather": {"station_id": "KTCM"},
        "training": {
            "schedule_seconds": 1800,
            "capture_interval_seconds": 300,
            "gradient_accumulation_steps": 4,
            "checkpoint_dir": "checkpoints",
            "lora": {"rank": 4, "alpha": 8, "target_modules": ["fc1"]},
        },
        "collection": {"collection_seconds": 600},
    }
    with open(config_file, "wb") as f:
        f.write(tomli_w.dumps(config_data).encode())

    loader = ConfigLoader(str(config_file))

    assert loader.webcam_url == "http://rainier-webcam.jpg"
    assert loader.metar_station == "KTCM"
    assert loader.mountain_data["name"] == "Mount Rainier"
    assert loader.schedule_seconds == 1800
    assert loader.collection_seconds == 600
    assert loader.lora_settings["rank"] == 4


def _write(tmp_path, extra):
    """Minimal valid config plus whatever section the test is about."""
    config_file = tmp_path / "mountain.toml"
    data = {
        "mountain": {"name": "Mount Rainier"},
        "webcam": {"url": "http://cam.jpg", "era": "king5-queenanne"},
        "weather": {"station_id": "KSEA"},
        "training": {"checkpoint_dir": "train/checkpoints"},
        "collection": {"collection_seconds": 600},
        **extra,
    }
    config_file.write_bytes(tomli_w.dumps(data).encode())
    return ConfigLoader(str(config_file))


def test_paths_and_mini_default_without_the_sections(tmp_path, monkeypatch):
    """A config predating [paths]/[mini] still resolves — no KeyError."""
    monkeypatch.setenv("HOME", "/Users/nobody")
    loader = _write(tmp_path, {})

    assert loader.paths["labels_dir"] == "/Users/nobody/mountain/data"
    assert loader.paths["checkpoint_dir"] == "/Users/nobody/mountain/checkpoints"
    assert loader.mini["host"].endswith(".local")
    # Derived from root, so two lines of config are enough.
    assert loader.mini["captures_dir"] == loader.mini["root"] + "/data"


def test_paths_expand_locally_and_mini_paths_do_not(tmp_path, monkeypatch):
    """`~` resolves for the local tree only.

    The [mini] values name directories on the far end of an rsync; expanding
    them against this machine's home would silently point at the wrong place.
    """
    monkeypatch.setenv("HOME", "/Users/tommy")
    loader = _write(
        tmp_path,
        {
            "paths": {
                "captures_dir": "~/mountain/data/",
                "labels_dir": "~/mountain/data",
                "checkpoint_dir": "~/mountain/checkpoints",
            },
            "mini": {
                "host": "tommydoerr@tommys-mac-mini.local",
                "root": "/Users/tommydoerr/mountain/",
            },
        },
    )

    # Expanded, and the trailing slash dropped so "$dir/x" never doubles up.
    assert loader.paths["captures_dir"] == "/Users/tommy/mountain/data"
    assert loader.mini["root"] == "/Users/tommydoerr/mountain"
    assert loader.mini["checkpoint_dir"] == "/Users/tommydoerr/mountain/checkpoints"
    assert "tommy/" not in loader.mini["checkpoint_dir"]


def test_mini_paths_can_be_overridden(tmp_path):
    loader = _write(
        tmp_path,
        {"mini": {"root": "/srv/mountain", "checkpoint_dir": "/srv/weights"}},
    )

    assert loader.mini["checkpoint_dir"] == "/srv/weights"
    assert loader.mini["labels_dir"] == "/srv/mountain/data"
    assert loader.mini["tick_label"] == "com.robogeosociety.mountain-tick"

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

from unittest.mock import MagicMock, patch
from click.testing import CliRunner
from collect.collector import once, live
from datetime import datetime, UTC


def run_sync_thread(target, daemon=True):
    """Mocks threading.Thread to run the target synchronously."""
    mock_thread = MagicMock()

    def start_mock():
        target()

    mock_thread.start = start_mock
    return mock_thread


@patch("collect.collector.ConfigLoader")
@patch("collect.collector.WebcamStream")
@patch("collect.collector.WeatherFetcher")
@patch("threading.Thread", side_effect=run_sync_thread)
def test_once_execution_utc_structure(
    mock_thread, mock_weather_cls, mock_webcam, mock_config, tmp_path
):
    """Verify that once function uses the new UTC-based directory structure."""
    mock_config.return_value.webcam_url = "Cam_JPG"
    mock_config.return_value.metar_station = "KSEA"
    mock_config.return_value.collection_seconds = 600

    mock_weather = MagicMock()
    mock_weather.fetch_latest_metar.return_value = "METAR KSEA 211953Z CLR 10/05"
    mock_weather_cls.return_value = mock_weather

    mock_stream = MagicMock()
    mock_stream.capture_raw.return_value = MagicMock()
    mock_webcam.return_value = mock_stream

    runner = CliRunner()
    # We also need to mock rumps.App.run to not block
    with patch("rumps.App.run"):
        with patch("cv2.imwrite") as mock_imwrite:
            # Use CliRunner to call the command with arguments
            result = runner.invoke(
                once,
                [
                    "--config",
                    "mountain.toml",
                    "--data-root",
                    str(tmp_path),
                    "--session-id",
                    "test-once",
                ],
            )

            assert result.exit_code == 0

            # Structure: data/YYYYMMDD/HHMMSS_UTC/
            now_utc = datetime.now(UTC)
            date_str = now_utc.strftime("%Y%m%d")

            date_dir = tmp_path / date_str
            assert date_dir.exists()

            time_dirs = list(date_dir.iterdir())
            assert len(time_dirs) > 0
            time_dir = time_dirs[0]
            assert time_dir.name.endswith("_UTC")
            assert (time_dir / "images").exists()
            assert (time_dir / "metar").exists()
            assert (time_dir / "metar" / "metar.txt").exists()
            assert mock_imwrite.called


@patch("collect.collector.ConfigLoader")
@patch("collect.collector.WebcamStream")
@patch("collect.collector.WeatherFetcher")
@patch("time.sleep", side_effect=KeyboardInterrupt)
def test_live_collection_loop(
    mock_sleep, mock_weather_cls, mock_webcam, mock_config, tmp_path
):
    """Verify that live command runs a loop and stops on KeyboardInterrupt."""
    mock_config.return_value.webcam_url = "Cam_JPG"
    mock_config.return_value.metar_station = "KSEA"
    mock_config.return_value.collection_seconds = 1

    mock_weather = MagicMock()
    mock_weather.fetch_latest_metar.return_value = "METAR KSEA"
    mock_weather_cls.return_value = mock_weather

    mock_stream = MagicMock()
    mock_stream.capture_raw.return_value = MagicMock()
    mock_webcam.return_value = mock_stream

    runner = CliRunner()
    with patch("cv2.imwrite"):
        # This will run one iteration and then 'sleep' will raise KeyboardInterrupt
        runner.invoke(live, ["--config", "mountain.toml", "--data-root", str(tmp_path)])

        # Check if directories were created for the first run
        now_utc = datetime.now(UTC)
        date_str = now_utc.strftime("%Y%m%d")
        assert (tmp_path / date_str).exists()

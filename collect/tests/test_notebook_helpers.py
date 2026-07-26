import json
import pytest
from unittest.mock import patch
from collect.notebook_helpers import (
    get_job_captures,
    get_job_status,
    CaptureBrowser,
    get_directory_captures,
    get_upcoming_schedule,
)


@pytest.fixture
def temp_log(tmp_path):
    log_file = tmp_path / "collection.log"
    return log_file


def test_get_job_captures_empty(temp_log):
    assert get_job_captures(temp_log) == []


def test_get_job_captures_valid(temp_log):
    entry = {
        "timestamp": "2026-02-23T12:00:00Z",
        "event": "CAPTURE",
        "status": "SUCCESS",
        "metadata": {
            "image_path": "data/test.jpg",
            "step_index": 1,
            "metar_path": "data/metar.txt",
        },
    }
    temp_log.write_text(json.dumps(entry) + "\n")

    captures = get_job_captures(temp_log)
    assert len(captures) == 1
    assert captures[0]["step"] == 1
    assert captures[0]["path"] == "data/test.jpg"


def test_get_job_status_valid(temp_log):
    entry = {
        "event": "PROGRESS",
        "status": "STATUS",
        "metadata": {"progress": 5, "total": 10, "percentage": 50.0},
    }
    temp_log.write_text(json.dumps(entry) + "\n")

    status = get_job_status(temp_log)
    assert status["progress"] == 5
    assert status["total"] == 10


def test_get_job_status_missing(temp_log):
    assert get_job_status(temp_log) is None


def test_get_directory_captures(tmp_path):
    # Create mock structure
    date_dir = tmp_path / "20260223"
    date_dir.mkdir()
    cap_dir = date_dir / "120000"
    cap_dir.mkdir()
    img_dir = cap_dir / "images"
    img_dir.mkdir()
    (img_dir / "test.jpg").write_text("fake image")

    metar_dir = cap_dir / "metar"
    metar_dir.mkdir()
    (metar_dir / "metar.txt").write_text("KSEA 231200Z")

    caps = get_directory_captures(tmp_path)
    assert len(caps) == 1
    assert caps[0]["metar"] is not None
    assert "test" in caps[0]["path"]


@patch("collect.notebook_helpers.widgets")
@patch("collect.notebook_helpers.display")
def test_capture_browser_init(mock_display, mock_widgets, temp_log):
    browser = CaptureBrowser(log_path=str(temp_log))
    assert browser.log_path == str(temp_log)
    assert browser.all_captures == []


@patch("collect.notebook_helpers.widgets")
@patch("collect.notebook_helpers.display")
@patch("collect.notebook_helpers.plt")
def test_render_control_panel(mock_plt, mock_display, mock_widgets, temp_log):
    browser = CaptureBrowser(log_path=str(temp_log))
    browser.render_control_panel()
    assert mock_widgets.VBox.called
    # It now sets children instead of calling display
    assert browser.control_container.children != []


@patch("pathlib.Path.exists")
def test_get_upcoming_schedule_missing(mock_exists):
    # Mock files as NOT existing
    mock_exists.return_value = False
    assert get_upcoming_schedule() == []

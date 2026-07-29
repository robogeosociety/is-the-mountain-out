"""Tests for collect.notebook_helpers — stubs ipywidgets so no kernel is needed."""

import json

import pytest
import yaml

from collect.notebook_helpers import CaptureBrowser, load_labels, save_labels

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _Widget:
    """Stand-in for any ipywidgets widget: accepts anything, records attributes."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.__dict__.update(kwargs)
        self.children = []

    def on_click(self, _callback):
        pass


class _WidgetsStub:
    """Returns a fresh `_Widget` for every `widgets.X(...)` lookup."""

    def __getattr__(self, _name):
        return _Widget


@pytest.fixture()
def log_path(tmp_path):
    return tmp_path / "collection.log"


@pytest.fixture()
def browser(log_path, tmp_path, monkeypatch):
    """A CaptureBrowser with ipywidgets/display stubbed out."""
    monkeypatch.setattr("collect.notebook_helpers.widgets", _WidgetsStub())
    monkeypatch.setattr("collect.notebook_helpers.display", lambda *_: None)
    return CaptureBrowser(log_path=str(log_path), data_root=str(tmp_path / "data"))


def write_log(path, *entries):
    """Writes JSONL entries (dicts, or raw strings for malformed lines)."""
    lines = [e if isinstance(e, str) else json.dumps(e) for e in entries]
    path.write_text("\n".join(lines) + "\n")


def capture_entry(timestamp, image_path, metar_path="data/metar.txt"):
    return {
        "timestamp": timestamp,
        "event": "CAPTURE",
        "status": "SUCCESS",
        "metadata": {"image_path": image_path, "metar_path": metar_path},
    }


# ---------------------------------------------------------------------------
# labels.yaml round-trip
# ---------------------------------------------------------------------------


def test_load_labels_missing_file(tmp_path):
    assert load_labels(tmp_path) == {}


def test_load_labels_empty_file(tmp_path):
    (tmp_path / "labels.yaml").write_text("")
    assert load_labels(tmp_path) == {}


def test_save_then_load_labels_roundtrip(tmp_path):
    labels = {"20260223/120000_us_UTC/images/a.jpg": 1, "b.jpg": 0}
    save_labels(tmp_path, labels)
    assert load_labels(tmp_path) == labels


def test_save_labels_writes_yaml(tmp_path):
    save_labels(tmp_path, {"a.jpg": 2})
    assert yaml.safe_load((tmp_path / "labels.yaml").read_text()) == {"a.jpg": 2}


# ---------------------------------------------------------------------------
# CaptureBrowser construction
# ---------------------------------------------------------------------------


def test_init_stores_paths_and_defaults(browser, log_path):
    assert browser.log_path == log_path
    assert browser.batch_size == 20
    assert browser.all_captures == []
    assert browser.current_page == 0


# ---------------------------------------------------------------------------
# CaptureBrowser._load_captures
# ---------------------------------------------------------------------------


def test_load_captures_missing_log(browser):
    assert browser._load_captures() == []


def test_load_captures_reads_successful_capture(browser, log_path):
    write_log(log_path, capture_entry("2026-02-23T12:00:00Z", "data/test.jpg"))

    assert browser._load_captures() == [
        {
            "timestamp": "2026-02-23T12:00:00Z",
            "image": "data/test.jpg",
            "metar": "data/metar.txt",
        }
    ]


def test_load_captures_metar_is_optional(browser, log_path):
    entry = capture_entry("2026-02-23T12:00:00Z", "data/test.jpg")
    del entry["metadata"]["metar_path"]
    write_log(log_path, entry)

    assert browser._load_captures()[0]["metar"] is None


def test_load_captures_sorts_newest_first(browser, log_path):
    write_log(
        log_path,
        capture_entry("2026-02-23T12:00:00Z", "old.jpg"),
        capture_entry("2026-02-23T14:00:00Z", "new.jpg"),
        capture_entry("2026-02-23T13:00:00Z", "mid.jpg"),
    )

    assert [c["image"] for c in browser._load_captures()] == [
        "new.jpg",
        "mid.jpg",
        "old.jpg",
    ]


def test_load_captures_ignores_other_events_and_failures(browser, log_path):
    failed = capture_entry("2026-02-23T12:00:00Z", "failed.jpg")
    failed["status"] = "FAILURE"
    progress = capture_entry("2026-02-23T12:30:00Z", "progress.jpg")
    progress["event"] = "PROGRESS"
    write_log(
        log_path, failed, progress, capture_entry("2026-02-23T13:00:00Z", "ok.jpg")
    )

    assert [c["image"] for c in browser._load_captures()] == ["ok.jpg"]


def test_load_captures_skips_malformed_lines(browser, log_path):
    write_log(
        log_path,
        "not json at all",
        {"event": "CAPTURE", "status": "SUCCESS"},  # no metadata key
        capture_entry("2026-02-23T13:00:00Z", "ok.jpg"),
    )

    assert [c["image"] for c in browser._load_captures()] == ["ok.jpg"]


# ---------------------------------------------------------------------------
# CaptureBrowser.refresh_ui
# ---------------------------------------------------------------------------


def test_refresh_ui_empty_log_shows_placeholder(browser):
    browser.refresh_ui()

    assert browser.all_captures == []
    assert "<b>Total Captures:</b> 0" in browser.status_label.value
    assert len(browser.grid_container.children) == 1


def test_refresh_ui_renders_one_tile_per_capture(browser, log_path, tmp_path):
    image = tmp_path / "shot.jpg"
    image.write_bytes(b"\xff\xd8\xff not-really-a-jpeg")
    metar = tmp_path / "metar.txt"
    metar.write_text("KSEA 231200Z 10SM CLR\n")
    write_log(
        log_path,
        capture_entry("2026-02-23T12:00:00Z", str(image), str(metar)),
        capture_entry("2026-02-23T13:00:00Z", str(image), str(metar)),
    )

    browser.refresh_ui()

    assert len(browser.all_captures) == 2
    assert len(browser.grid_container.children) == 2
    assert "<b>Showing:</b> 1-2" in browser.status_label.value


def test_refresh_ui_skips_captures_whose_image_is_gone(browser, log_path, tmp_path):
    image = tmp_path / "shot.jpg"
    image.write_bytes(b"\xff\xd8\xff not-really-a-jpeg")
    write_log(
        log_path,
        capture_entry("2026-02-23T12:00:00Z", str(image), None),
        capture_entry("2026-02-23T13:00:00Z", str(tmp_path / "vanished.jpg"), None),
    )

    browser.refresh_ui()

    # Both captures are indexed; only the readable one is rendered.
    assert len(browser.all_captures) == 2
    assert len(browser.grid_container.children) == 1


def test_refresh_ui_paginates_by_batch_size(browser, log_path, tmp_path):
    image = tmp_path / "shot.jpg"
    image.write_bytes(b"\xff\xd8\xff not-really-a-jpeg")
    browser.batch_size = 2
    write_log(
        log_path,
        *[capture_entry(f"2026-02-23T1{i}:00:00Z", str(image), None) for i in range(5)],
    )

    browser.refresh_ui()
    assert len(browser.grid_container.children) == 2
    assert "<b>Showing:</b> 1-2" in browser.status_label.value

    browser.current_page = 2
    browser.refresh_ui()
    assert len(browser.grid_container.children) == 1
    assert "<b>Showing:</b> 5-5" in browser.status_label.value

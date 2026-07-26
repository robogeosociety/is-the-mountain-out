"""Tests for MountainTray — mocks rumps so no GUI event loop is needed."""

import pytest

from collect.state import CollectorState, write_state, read_state, make_state
from collect.tray import MountainTray, _fmt_time  # noqa: E402
from collect.collector import _derive_initial_last_capture_at


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def data_root(tmp_path):
    return tmp_path


@pytest.fixture()
def tray(data_root):
    return MountainTray(data_root=str(data_root), session_id="abc123")


@pytest.fixture()
def base_state():
    return CollectorState(
        session_id="abc123",
        status="Idle",
        capture_count=10,
        plan_total=628,
        interval_seconds=600,
        last_capture_at="2026-03-15T05:44:16+00:00",
        next_capture_at="2026-03-15T05:54:16+00:00",
        label_counts={"0": 100, "1": 5, "2": 20},
        updated_at="2026-03-15T05:44:17+00:00",
    )


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_initial_title_is_mountain_emoji(tray):
    assert tray.title == "🗻"


def test_initial_status_placeholder(tray):
    assert "—" in tray.status_item.title


# ---------------------------------------------------------------------------
# State file read/write round-trip
# ---------------------------------------------------------------------------


def test_write_and_read_state_roundtrip(data_root, base_state):
    write_state(data_root, base_state)
    result = read_state(data_root, base_state.session_id)
    assert result == base_state


def test_read_state_returns_none_when_missing(data_root):
    assert read_state(data_root, "missing_session") is None


def test_write_is_atomic(data_root, base_state):
    """write_state uses a tmp file + rename — no partial reads."""
    write_state(data_root, base_state)
    path = data_root / f"collector_state_{base_state.session_id}.json"
    assert path.exists()
    # No .tmp file should remain
    assert not (data_root / f"collector_state_{base_state.session_id}.tmp").exists()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_populates_status(tray, base_state):
    tray._render(base_state)
    assert "Idle" in tray.status_item.title


def test_render_populates_progress(tray, base_state):
    tray._render(base_state)
    assert "10/628" in tray.progress_item.title
    assert "1%" in tray.progress_item.title  # 10/628 = 1%


def test_render_populates_next_capture(tray, base_state):
    tray._render(base_state)
    assert "Next Capture:" in tray.next_item.title
    assert "—" not in tray.next_item.title


def test_render_populates_session(tray, base_state):
    tray._render(base_state)
    assert "abc123" in tray.session_item.title


def test_render_shows_placeholder_when_next_is_none(tray, base_state):
    base_state.next_capture_at = None
    tray._render(base_state)
    assert "—" in tray.next_item.title


def test_render_spinner_cycles(tray, base_state):
    from collect.tray import SPINNER

    base_state.status = "capturing"

    # First frame
    tray._render(base_state)
    assert tray.title == f"🗻{SPINNER[0]}"

    # Second frame
    tray._render(base_state)
    assert tray.title == f"🗻{SPINNER[1]}"

    # Wrap around (after len(SPINNER) calls)
    tray._spinner_idx = len(SPINNER) - 1
    tray._render(base_state)
    assert tray.title == f"🗻{SPINNER[-1]}"
    tray._render(base_state)
    assert tray.title == f"🗻{SPINNER[0]}"


# ---------------------------------------------------------------------------
# _refresh reads from state file
# ---------------------------------------------------------------------------


def test_refresh_reads_state_file(tray, data_root, base_state):
    write_state(data_root, base_state)
    tray._refresh()
    assert "Idle" in tray.status_item.title
    assert "abc123" in tray.session_item.title


def test_refresh_shows_error_when_no_state_file(tray):
    tray._refresh()
    assert "No state found" in tray.status_item.title


def test_refresh_skips_rerender_when_state_unchanged(tray, data_root, base_state):
    write_state(data_root, base_state)
    tray._refresh()
    # Mutate a menu item directly — a re-render would reset it
    tray.status_item.title = "SENTINEL"
    tray._refresh()  # same state object, should skip
    assert tray.status_item.title == "SENTINEL"


# ---------------------------------------------------------------------------
# pct_complete
# ---------------------------------------------------------------------------


def test_pct_complete_normal():
    s = make_state("s", "Idle", capture_count=314, interval_seconds=600, plan_total=628)
    assert s.pct_complete == 50


def test_pct_complete_caps_at_100():
    s = make_state("s", "Idle", capture_count=999, interval_seconds=600, plan_total=628)
    assert s.pct_complete == 100


def test_pct_complete_zero():
    s = make_state("s", "Idle", capture_count=0, interval_seconds=600, plan_total=628)
    assert s.pct_complete == 0


def test_pct_complete_unknown_plan():
    s = make_state("s", "Idle", capture_count=10, interval_seconds=600, plan_total=0)
    assert s.pct_complete == 0


# ---------------------------------------------------------------------------
# _fmt_time helper
# ---------------------------------------------------------------------------


def test_fmt_time_past_date_includes_month_and_day():
    # A date in the past — should show "Mar 15 HH:MM" style
    result = _fmt_time("2026-03-15T05:44:16+00:00")
    assert result is not None
    assert ":" in result
    # Either "Today HH:MM" or "Mon DD HH:MM" — both contain a space and colon
    assert " " in result


def test_fmt_time_future_date_includes_month_and_day():
    result = _fmt_time("2026-04-10T13:00:00+00:00")
    assert result is not None
    assert "Apr" in result or "Today" in result


def test_fmt_time_none():
    assert _fmt_time(None) is None


def test_fmt_time_invalid():
    assert _fmt_time("not-a-date") is None


# ---------------------------------------------------------------------------
# last_capture_at preservation through capture loop state writes
# ---------------------------------------------------------------------------


def test_derive_initial_last_capture_at_from_plan_past(tmp_path):
    """Returns most recent past plan timestamp when one exists."""
    past = "2026-03-14T20:00:00+00:00"
    future = "2026-03-15T20:00:00+00:00"
    from datetime import datetime, timezone

    now = datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc)
    result = _derive_initial_last_capture_at([past, future], str(tmp_path), now, "sess")
    assert result == past


def test_derive_initial_last_capture_at_falls_back_to_state_file(tmp_path):
    """When plan has no past timestamps, uses previous state file's last_capture_at."""
    past_time = "2026-03-14T20:00:00+00:00"
    write_state(
        tmp_path,
        make_state(
            "prev",
            "Idle",
            capture_count=5,
            interval_seconds=600,
            plan_total=636,
            last_capture_at=past_time,
        ),
    )
    future = "2026-03-15T20:00:00+00:00"
    from datetime import datetime, timezone

    now = datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc)
    result = _derive_initial_last_capture_at([future], str(tmp_path), now, "prev")
    assert result == past_time


def test_derive_initial_last_capture_at_ignores_future_state(tmp_path):
    """Does not use previous state's last_capture_at if it's in the future."""
    future_time = "2026-03-16T20:00:00+00:00"
    write_state(
        tmp_path,
        make_state(
            "prev",
            "Idle",
            capture_count=5,
            interval_seconds=600,
            plan_total=636,
            last_capture_at=future_time,
        ),
    )
    from datetime import datetime, timezone

    now = datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc)
    result = _derive_initial_last_capture_at([], str(tmp_path), now, "prev")
    assert result is None


def test_derive_initial_last_capture_at_no_plan_no_state(tmp_path):
    from datetime import datetime, timezone

    now = datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc)
    assert _derive_initial_last_capture_at([], str(tmp_path), now, "any") is None


def test_make_state_resets_last_capture_at_when_omitted(tmp_path):
    """Regression guard: make_state without last_capture_at yields None — callers must pass it."""
    state = make_state(
        "sess1", "Capturing...", capture_count=0, interval_seconds=600, plan_total=5
    )
    assert state.last_capture_at is None  # documenting the footgun the loop must avoid


def test_capturing_state_preserves_last_capture_at(tmp_path):
    """The capture loop must pass last_capture_at to EVERY write_state call, including Capturing."""
    past_time = "2026-03-14T20:00:00+00:00"
    # Simulate the fixed loop: Capturing... write includes last_capture_at explicitly
    write_state(
        tmp_path,
        make_state(
            "sess1",
            "Capturing...",
            capture_count=0,
            interval_seconds=600,
            plan_total=5,
            last_capture_at=past_time,
        ),
    )
    state = read_state(tmp_path, "sess1")
    assert state.last_capture_at == past_time

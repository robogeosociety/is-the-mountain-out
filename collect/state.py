"""
Collector state file — written by the capture service, read by the tray.

The state file lives at {data_root}/collector_state.json and can be
updated by any process that has access to the data directory, independently
of whether the capture service is running.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

STATE_FILENAME = "collector_state.json"
PLAN_FILENAME = "capture_plan.json"


def _get_state_path(data_root: str | Path, session_id: str) -> Path:
    return Path(data_root) / f"collector_state_{session_id}.json"


@dataclass
class CollectorState:
    session_id: str
    status: str  # "Idle", "Capturing...", "Error"
    capture_count: int
    plan_total: int  # total planned captures (0 = unknown)
    interval_seconds: int
    last_capture_at: Optional[str]  # ISO-8601 UTC
    next_capture_at: Optional[str]  # ISO-8601 UTC
    label_counts: Dict[str, int]  # {"0": N, "1": N, "2": N}
    updated_at: str  # ISO-8601 UTC
    session_labels_file: Optional[str] = None  # path to labels.{uuid}.yaml
    final_capture_at: Optional[str] = None  # ISO-8601 UTC of last planned capture
    prior_capture_count: int = 0  # captures from previous sessions toward same goal
    prior_plan_total: int = 0  # plan_total from previous sessions toward same goal

    @property
    def pct_complete(self) -> int:
        total = self.plan_total + self.prior_plan_total
        done = self.capture_count + self.prior_capture_count
        if total <= 0:
            return 0
        return min(100, int(done / total * 100))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(data_root: str | Path, state: CollectorState) -> None:
    path = _get_state_path(data_root, state.session_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2))
    tmp.replace(path)  # atomic on POSIX


def read_state(data_root: str | Path, session_id: str) -> Optional[CollectorState]:
    path = _get_state_path(data_root, session_id)
    try:
        data = json.loads(path.read_text())
        return CollectorState(**data)
    except Exception:
        return None


def write_plan(data_root: str | Path, timestamps: list[str]) -> Path:
    """Save a list of ISO-8601 UTC capture timestamps to capture_plan.json."""
    path = Path(data_root) / PLAN_FILENAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {
                "generated_at": _now_iso(),
                "total": len(timestamps),
                "captures": timestamps,
            },
            indent=2,
        )
    )
    tmp.replace(path)
    return path


def read_plan(data_root: str | Path) -> Optional[list[str]]:
    """Return list of ISO-8601 UTC capture timestamps, or None if no plan file."""
    path = Path(data_root) / PLAN_FILENAME
    try:
        data = json.loads(path.read_text())
        return data["captures"]
    except Exception:
        return None


def read_label_counts(data_root: str | Path) -> Dict[str, int]:
    """Reads labels.yaml and returns per-class counts as string keys."""
    labels_path = Path(data_root) / "labels.yaml"
    if not labels_path.exists():
        return {}
    try:
        import yaml

        with open(labels_path) as f:
            labels = yaml.safe_load(f) or {}
        counts: Dict[str, int] = {}
        for v in labels.values():
            key = str(v)
            counts[key] = counts.get(key, 0) + 1
        return counts
    except Exception:
        return {}


def make_state(
    session_id: str,
    status: str,
    capture_count: int,
    interval_seconds: int,
    plan_total: int = 0,
    last_capture_at: Optional[str] = None,
    next_capture_at: Optional[str] = None,
    label_counts: Optional[Dict[str, int]] = None,
    session_labels_file: Optional[str] = None,
    final_capture_at: Optional[str] = None,
    prior_capture_count: int = 0,
    prior_plan_total: int = 0,
) -> CollectorState:
    return CollectorState(
        session_id=session_id,
        status=status,
        capture_count=capture_count,
        plan_total=plan_total,
        interval_seconds=interval_seconds,
        last_capture_at=last_capture_at,
        next_capture_at=next_capture_at,
        label_counts=label_counts or {},
        updated_at=_now_iso(),
        session_labels_file=session_labels_file,
        final_capture_at=final_capture_at,
        prior_capture_count=prior_capture_count,
        prior_plan_total=prior_plan_total,
    )

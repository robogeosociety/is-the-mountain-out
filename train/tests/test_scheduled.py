"""Tests for train.scheduled — the gate, watermark, and telemetry embeds.

Pure logic only: no subprocesses, no network, LocalStorage for the watermark
(mirroring bot/tests conventions).
"""

import json
from datetime import UTC, datetime

from collect.storage import LocalStorage
from train import scheduled


def _event(ts: str, capture: str = "20260726/x/images/a.jpg") -> dict:
    return {"timestamp": ts, "capture": capture, "label": 0}


class TestGate:
    def test_parse_skips_malformed_lines(self):
        text = (
            json.dumps(_event("2026-07-26T16:53:39+00:00"))
            + "\nnot-json\n"
            + json.dumps({"no_timestamp": True})
            + "\n"
            + json.dumps(_event("2026-07-27T10:00:00+00:00"))
        )
        assert len(scheduled.parse_events(text)) == 2

    def test_no_watermark_means_all_pending(self):
        events = [_event("2026-07-26T16:53:39+00:00")]
        assert scheduled.pending_events(events, None) == events

    def test_older_and_equal_events_are_not_pending(self):
        mark = "2026-07-26T16:53:39+00:00"
        events = [
            _event("2026-07-25T00:00:00+00:00"),
            _event(mark),
            _event("2026-07-27T00:00:00+00:00"),
        ]
        pending = scheduled.pending_events(events, mark)
        assert len(pending) == 1
        assert pending[0]["timestamp"] == "2026-07-27T00:00:00+00:00"

    def test_bad_watermark_fails_open(self):
        events = [_event("2026-07-26T16:53:39+00:00")]
        assert scheduled.pending_events(events, "garbage") == events


class TestWatermark:
    def test_round_trip(self, tmp_path):
        store = LocalStorage(str(tmp_path))
        assert scheduled.load_watermark(store) == {}
        mark = {"last_event_ts": "2026-07-26T16:53:39+00:00", "best_val_loss": 0.0782}
        scheduled.save_watermark(store, mark)
        assert scheduled.load_watermark(store) == mark

    def test_corrupt_watermark_is_empty(self, tmp_path):
        store = LocalStorage(str(tmp_path))
        store.put_text(scheduled.WATERMARK_KEY, "{not json")
        assert scheduled.load_watermark(store) == {}


SUMMARY = {
    "status": "ok",
    "labels_loaded": 2004,
    "class_counts": {"not_out": 1729, "full": 109, "partial": 166},
    "best_val_loss": 0.0765,
    "best_epoch": 3,
    "per_epoch": [
        {"epoch": 1, "train_loss": 0.31, "val_loss": 0.09, "val_acc": 0.955},
        {"epoch": 2, "train_loss": 0.22, "val_loss": 0.0765, "val_acc": 0.979},
    ],
    "checkpoint_keys_uploaded": [
        "checkpoints/adapter_config.json",
        "checkpoints/adapter_model.safetensors",
        "checkpoints/classifier.pt",
    ],
}


class TestEmbeds:
    def test_start_embed(self):
        embed = scheduled.start_embed(4, 5, datetime(2026, 7, 27, 11, tzinfo=UTC))
        assert embed["color"] == scheduled.COLOR_RUNNING
        assert "4 new Discord label event(s)" in embed["fields"][0]["value"]

    def test_result_embed_improvement(self):
        embed = scheduled.result_embed(SUMMARY, 4, previous_best=0.0782, duration_s=900)
        values = {f["name"]: f["value"] for f in embed["fields"]}
        assert embed["color"] == scheduled.COLOR_OK
        assert "▼ improved 0.0017" in values["Best val loss"]
        assert "1729/109/166" in values["Dataset"]
        assert values["Duration"] == "15 min"
        assert "3/3 files" in values["Checkpoint"]

    def test_result_embed_regression_flagged(self):
        worse = dict(SUMMARY, best_val_loss=0.0900)
        embed = scheduled.result_embed(worse, 1, previous_best=0.0782, duration_s=60)
        values = {f["name"]: f["value"] for f in embed["fields"]}
        assert "▲ REGRESSED" in values["Best val loss"]

    def test_result_embed_first_run(self):
        embed = scheduled.result_embed(SUMMARY, 2, previous_best=None, duration_s=60)
        values = {f["name"]: f["value"] for f in embed["fields"]}
        assert "first scheduled run" in values["Best val loss"]

    def test_result_embed_partial_upload_warns(self):
        partial = dict(SUMMARY, checkpoint_keys_uploaded=["checkpoints/classifier.pt"])
        embed = scheduled.result_embed(partial, 1, previous_best=None, duration_s=60)
        values = {f["name"]: f["value"] for f in embed["fields"]}
        assert embed["color"] == scheduled.COLOR_FAILED
        assert "only 1/3" in values["Checkpoint"]

    def test_failure_embed_keeps_watermark_promise(self):
        embed = scheduled.failure_embed(3, "training exit 1")
        assert embed["color"] == scheduled.COLOR_FAILED
        assert "watermark not advanced" in embed["fields"][2]["value"]

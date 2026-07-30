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
    "best_epoch": 2,
    "best_val_acc": 0.979,
    "prefetch_s": 372.0,
    "per_epoch": [
        {
            "epoch": 1,
            "train_loss": 0.31,
            "val_loss": 0.09,
            "val_acc": 0.955,
            "duration_s": 180.0,
        },
        {
            "epoch": 2,
            "train_loss": 0.22,
            "val_loss": 0.0765,
            "val_acc": 0.979,
            "duration_s": 192.0,
        },
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
        embed = scheduled.result_embed(
            SUMMARY, 4, previous_best=0.0782, duration_s=900, previous_best_acc=0.976
        )
        values = {f["name"]: f["value"] for f in embed["fields"]}
        assert embed["color"] == scheduled.COLOR_OK
        assert "▼ improved 0.0017" in values["Best val loss"]
        assert "1729/109/166" in values["Dataset"]
        assert "3/3 files" in values["Checkpoint"]
        # Time telemetry: total · prefetch · avg per-epoch.
        assert values["Duration"] == "15 min total · prefetch 6.2 min · ~3.1 min/epoch"
        # Accuracy improvement in percentage points, arrows flipped vs loss.
        assert values["Val accuracy"] == "97.9% — ▲ improved 0.3pt vs last (97.6%)"

    def test_acc_regression_and_first_run_text(self):
        assert "▼ regressed 1.9pt" in scheduled._acc_delta_text(0.957, 0.976)
        assert "no previous run" in scheduled._acc_delta_text(0.979, None)
        assert scheduled._acc_delta_text(None, 0.976) == "n/a"

    def test_result_embed_acc_fallback_from_per_epoch(self):
        # Older summaries lack best_val_acc — recover it from the best epoch's record.
        legacy = {k: v for k, v in SUMMARY.items() if k != "best_val_acc"}
        embed = scheduled.result_embed(legacy, 1, previous_best=None, duration_s=600)
        values = {f["name"]: f["value"] for f in embed["fields"]}
        assert values["Val accuracy"].startswith("97.9%")

    def test_duration_text_without_breakdown(self):
        assert scheduled._duration_text(600, {}) == "10 min total"

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


class TestMemoryTelemetry:
    def test_renders_macos_mps_probe(self):
        text = scheduled._memory_text(
            {"rss_peak_mb": 4300.0, "mps_allocated_mb": 1843.2, "mps_driver_mb": 2150.4}
        )
        assert "peak RSS 4.20 GB" in text
        assert "MPS 1.80 GB (2.10 GB driver)" in text

    def test_missing_probes_degrade_not_crash(self):
        assert scheduled._memory_text(None) == "n/a"
        assert scheduled._memory_text({}) == "n/a"
        # A CPU-only box has RSS but no accelerator keys.
        assert scheduled._memory_text({"rss_peak_mb": 2048.0}) == "peak RSS 2.00 GB"

    def test_peak_is_the_high_water_mark_across_epochs(self):
        peak = scheduled.peak_memory(
            [
                {"memory": {"rss_peak_mb": 1000.0, "mps_allocated_mb": 500.0}},
                {"memory": {"rss_peak_mb": 3000.0, "mps_allocated_mb": 400.0}},
                {},  # an epoch recorded before this feature existed
            ]
        )
        # Max is per key, independently: RSS peaks in epoch 2, MPS in epoch 1.
        assert peak == {"rss_peak_mb": 3000.0, "mps_allocated_mb": 500.0}


class TestCheckpointEmbed:
    RECORD = {
        "epoch": 3,
        "total_epochs": 5,
        "val_loss": 0.0163,
        "val_acc": 0.998,
        "duration_s": 594.0,
        "previous_best_val_loss": 0.0205,
        "checkpoint_keys": ["a", "b", "c"],
        "memory": {"rss_peak_mb": 4300.0},
        "checkpoint_saved": True,
    }

    def test_reports_improvement_over_the_run_best(self):
        embed = scheduled.checkpoint_embed(self.RECORD)
        values = {f["name"]: f["value"] for f in embed["fields"]}
        assert "epoch 3/5" in embed["title"]
        assert "0.0042 better than 0.0205" in values["Val loss"]
        assert values["Val accuracy"] == "99.8%"
        assert values["Epoch time"] == "9.9 min"
        assert "peak RSS 4.20 GB" in values["Memory"]
        assert embed["color"] == scheduled.COLOR_OK

    def test_first_checkpoint_has_nothing_to_compare(self):
        first = dict(self.RECORD, previous_best_val_loss=None)
        values = {
            f["name"]: f["value"] for f in scheduled.checkpoint_embed(first)["fields"]
        }
        assert "first checkpoint this run" in values["Val loss"]

    def test_partial_upload_is_flagged_red(self):
        partial = dict(self.RECORD, checkpoint_keys=["a"])
        embed = scheduled.checkpoint_embed(partial)
        assert embed["color"] == scheduled.COLOR_FAILED
        assert (
            "1/3 files" in {f["name"]: f["value"] for f in embed["fields"]}["Uploaded"]
        )


class _RecordingTelemetry:
    def __init__(self):
        self.posted = []

    def post_standalone(self, embed):
        self.posted.append(embed)


class TestDrainProgress:
    def test_only_checkpoint_lines_post(self, tmp_path):
        path = tmp_path / "progress.jsonl"
        path.write_text(
            json.dumps({"epoch": 1, "checkpoint_saved": False, "val_loss": 0.9})
            + "\n"
            + json.dumps(
                {
                    "epoch": 2,
                    "checkpoint_saved": True,
                    "val_loss": 0.5,
                    "total_epochs": 5,
                    "checkpoint_keys": ["a", "b", "c"],
                }
            )
            + "\n"
        )
        tel = _RecordingTelemetry()
        offset = scheduled.drain_progress(path, 0, tel)
        assert len(tel.posted) == 1
        assert "epoch 2/5" in tel.posted[0]["title"]
        assert offset == path.stat().st_size

    def test_resumes_from_offset_without_reposting(self, tmp_path):
        path = tmp_path / "progress.jsonl"
        line = (
            json.dumps(
                {
                    "epoch": 1,
                    "checkpoint_saved": True,
                    "total_epochs": 5,
                    "val_loss": 0.5,
                    "checkpoint_keys": ["a", "b", "c"],
                }
            )
            + "\n"
        )
        path.write_text(line)
        tel = _RecordingTelemetry()
        offset = scheduled.drain_progress(path, 0, tel)
        assert len(tel.posted) == 1
        # Nothing new appended -> no duplicate post.
        assert scheduled.drain_progress(path, offset, tel) == offset
        assert len(tel.posted) == 1

    def test_partial_trailing_line_is_left_for_the_next_poll(self, tmp_path):
        path = tmp_path / "progress.jsonl"
        complete = (
            json.dumps(
                {
                    "epoch": 1,
                    "checkpoint_saved": True,
                    "total_epochs": 5,
                    "val_loss": 0.5,
                    "checkpoint_keys": ["a", "b", "c"],
                }
            )
            + "\n"
        )
        path.write_bytes((complete + '{"epoch": 2, "checkpoint_sav').encode())
        tel = _RecordingTelemetry()
        offset = scheduled.drain_progress(path, 0, tel)
        assert len(tel.posted) == 1
        assert offset == len(complete.encode()), "partial line must not be consumed"

        # The writer finishes the line; the next poll picks it up exactly once.
        with open(path, "a") as handle:
            handle.write(
                'ed": true, "total_epochs": 5, "val_loss": 0.4, '
                '"epoch_done": 1, "checkpoint_keys": ["a","b","c"]}\n'
            )
        scheduled.drain_progress(path, offset, tel)
        assert len(tel.posted) == 2

    def test_missing_file_and_garbage_are_survivable(self, tmp_path):
        tel = _RecordingTelemetry()
        assert scheduled.drain_progress(tmp_path / "nope.jsonl", 0, tel) == 0
        path = tmp_path / "progress.jsonl"
        path.write_text("not json\n\n" + json.dumps({"checkpoint_saved": False}) + "\n")
        scheduled.drain_progress(path, 0, tel)
        assert tel.posted == []


class TestEmitterConsumerContract:
    """The trainer writes these lines and scheduled.py parses them. The two live
    in different modules and different processes, so agreement is asserted here
    rather than assumed."""

    def test_real_emitter_output_drains_into_a_checkpoint_post(
        self, tmp_path, monkeypatch
    ):
        import json as _json
        import os as _os

        progress = tmp_path / "progress.jsonl"

        # Reproduce train.scheduler.batch's _emit_progress verbatim in shape:
        # append one json line, flush, fsync.
        def emit(event):
            with open(progress, "a") as handle:
                handle.write(_json.dumps(event) + "\n")
                handle.flush()
                _os.fsync(handle.fileno())

        from train.scheduler import memory_snapshot

        memory = memory_snapshot()
        assert isinstance(memory, dict), "probe must always return a dict"

        emit(
            {
                "epoch": 1,
                "total_epochs": 5,
                "train_loss": 0.9,
                "val_loss": 0.8,
                "val_acc": 0.7,
                "duration_s": 300.0,
                "memory": memory,
                "checkpoint_saved": False,
            }
        )
        emit(
            {
                "epoch": 2,
                "total_epochs": 5,
                "train_loss": 0.5,
                "val_loss": 0.4,
                "val_acc": 0.95,
                "duration_s": 310.0,
                "memory": memory,
                "checkpoint_saved": True,
                "previous_best_val_loss": 0.8,
                "checkpoint_keys": [
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "classifier.pt",
                ],
            }
        )

        tel = _RecordingTelemetry()
        offset = scheduled.drain_progress(progress, 0, tel)

        assert offset == progress.stat().st_size
        assert len(tel.posted) == 1, "only the checkpoint epoch posts"
        embed = tel.posted[0]
        values = {f["name"]: f["value"] for f in embed["fields"]}
        assert "epoch 2/5" in embed["title"]
        assert "0.4000" in values["Val loss"]
        assert values["Uploaded"] == "3/3 files → R2"
        # Whatever this machine's probe returned must render, not explode.
        assert isinstance(values["Memory"], str) and values["Memory"]

    def test_batch_exposes_the_progress_flag(self):
        """--progress-jsonl must exist, or scheduled.py's subprocess call fails
        at runtime with an unrecognised-option error nothing else would catch."""
        import inspect

        from train import scheduler

        assert "progress_jsonl" in inspect.signature(scheduler.batch).parameters

"""Supervisor-scheduled weekly retrain — gate on new labels, train, report.

Registered as `mountain-weekly-train` in the robogeosociety/supervisor REGISTRY
(Mon 04:00, heavy). The cron only decides when to LOOK; this body decides
whether to RUN, following the fleet's cheap-no-op convention:

1. Read `labels/discord-events.jsonl` (R2 truth) against the watermark at
   `labels/train-watermark.json`; exit 0 quietly when nothing is new.
2. Refresh `data/labels.yaml` from R2 (`collect sync labels pull`), post a
   start embed to #mountain (bot-token REST — no gateway needed), then run
   `training batch --json-summary --progress-jsonl` and trust those files, not
   stdout.
3. While it runs, tail the progress jsonl and post each SAVED CHECKPOINT as its
   own message (~3 per 5-epoch run — only improvements save), with per-epoch
   metrics and memory.
4. Success → advance the watermark and edit the start embed with final metrics
   (best-val-loss delta vs the previous run, peak memory). Failure → edit it
   with the error and leave the watermark, so next week retries the same delta.

Run via scripts/scheduled-train.sh (sources cf.env). Discord being down never
fails a training run — telemetry is best-effort.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import requests

from bot.labeler import EVENTS_KEY, uncached_storage
from train.config_loader import ConfigLoader
from train.metrics import CLASS_LABELS, CLASS_NAMES, format_confusion

WATERMARK_KEY = "labels/train-watermark.json"

COLOR_RUNNING = 0x3498DB
COLOR_OK = 0x2ECC71
COLOR_FAILED = 0xE74C3C

DISCORD_API = "https://discord.com/api/v10"


# ---------- Gate ----------


def parse_events(events_text: str) -> list[dict]:
    """Parse the provenance jsonl; malformed lines are skipped, never fatal."""
    events = []
    for line in events_text.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and "timestamp" in event:
            events.append(event)
    return events


def pending_events(events: list[dict], watermark_ts: str | None) -> list[dict]:
    """Events strictly newer than the watermark (all of them when no watermark)."""
    if not watermark_ts:
        return list(events)
    try:
        mark = datetime.fromisoformat(watermark_ts)
    except ValueError:
        return list(events)
    out = []
    for event in events:
        try:
            if datetime.fromisoformat(event["timestamp"]) > mark:
                out.append(event)
        # Keep the parens 3.13-compatible; ruff's py314 style would strip them.
        except (ValueError, TypeError):  # fmt: skip
            continue
    return out


def load_watermark(storage) -> dict:
    try:
        return json.loads(storage.get_text(WATERMARK_KEY))
    except Exception:  # noqa: BLE001 — miss/IO/parse all mean "never trained on events"
        return {}


def save_watermark(storage, watermark: dict) -> None:
    storage.put_text(WATERMARK_KEY, json.dumps(watermark, indent=2))


# ---------- Telemetry embeds ----------


def start_embed(pending_n: int, epochs: int, started_at: datetime) -> dict:
    return {
        "title": "🧠 Weekly training run",
        "color": COLOR_RUNNING,
        "fields": [
            {
                "name": "Trigger",
                "value": f"{pending_n} new Discord label event(s) since last run",
                "inline": False,
            },
            {
                "name": "Status",
                "value": f"training… ({epochs} epochs)",
                "inline": False,
            },
        ],
        "footer": {"text": "is-the-mountain-out • scheduled training"},
        "timestamp": started_at.isoformat(),
    }


def _delta_text(best: float, previous: float | None) -> str:
    if previous is None:
        return "first scheduled run"
    delta = best - previous
    if abs(delta) < 1e-6:
        return f"unchanged vs last ({previous:.4f})"
    arrow = "▼ improved" if delta < 0 else "▲ REGRESSED"
    return f"{arrow} {abs(delta):.4f} vs last ({previous:.4f})"


def _acc_delta_text(acc: float | None, previous: float | None) -> str:
    """Accuracy improvement in percentage points (higher is better — arrows flip
    relative to the loss delta)."""
    if acc is None:
        return "n/a"
    if previous is None:
        return f"{acc:.1%} — no previous run to compare"
    delta_pt = (acc - previous) * 100
    if abs(delta_pt) < 0.05:
        return f"{acc:.1%} — unchanged vs last"
    arrow = "▲ improved" if delta_pt > 0 else "▼ regressed"
    return f"{acc:.1%} — {arrow} {abs(delta_pt):.1f}pt vs last ({previous:.1%})"


def _f1_delta_text(macro_f1: float | None, previous: float | None) -> str:
    """Macro-F1 with a run-over-run delta.

    Macro-F1 is the headline because accuracy is not one: on this label set
    (86% Not Out) a model that never says "out" scores 86% accuracy and 0.31
    macro-F1. Higher is better, so arrows match `_acc_delta_text`, not the loss
    delta. Reported to 3 decimals — 2 hides the movement small val classes make.
    """
    if macro_f1 is None:
        return "n/a — run predates per-class metrics"
    if previous is None:
        return f"{macro_f1:.3f} — no previous run to compare"
    delta = macro_f1 - previous
    if abs(delta) < 0.0005:
        return f"{macro_f1:.3f} — unchanged vs last"
    arrow = "▲ improved" if delta > 0 else "▼ regressed"
    return f"{macro_f1:.3f} — {arrow} {abs(delta):.3f} vs last ({previous:.3f})"


def _recall_text(metrics: Mapping[str, Any] | None) -> str:
    """Per-class recall + precision for the two classes anyone cares about.

    Not Out is included last for completeness, but Full and Partial lead: they
    are 5.5% and 8.2% of the data, and they are the entire product. `n=` is the
    val support, printed on purpose — at n≈17 one flipped frame is ~6pt of
    recall, and a reader who cannot see the denominator will over-read the move.
    """
    if not metrics:
        return "n/a — run predates per-class metrics"
    per_class = metrics.get("per_class") or {}
    lines = []
    for name, label in zip(CLASS_NAMES, CLASS_LABELS, strict=True):
        stats = per_class.get(name)
        if not stats:
            continue
        if not stats.get("support"):
            lines.append(f"**{label}** — absent from val set")
            continue
        lines.append(
            f"**{label}** recall {float(stats['recall']):.0%} · "
            f"precision {float(stats['precision']):.0%} (n={stats['support']})"
        )
    return "\n".join(lines) or "n/a"


def _visible_text(metrics: Mapping[str, Any] | None) -> str:
    """The product question: Full+Partial collapsed against Not Out.

    Precision leads because the repo's stated constraint is precision over
    recall — a false positive is an alert claiming the mountain is out when it
    is not, which is the failure users actually notice.
    """
    if not metrics:
        return "n/a — run predates per-class metrics"
    visible = metrics.get("visible")
    if not visible:
        return "n/a"
    return (
        f"precision {float(visible['precision']):.0%} · "
        f"recall {float(visible['recall']):.0%} "
        f"(n={visible.get('support', '?')} visible frames in val)"
    )


def _confusion_text(metrics: Mapping[str, Any] | None) -> str:
    """A code block, not nine embed fields — 9 fields would eat the embed."""
    if not metrics or not metrics.get("confusion"):
        return "n/a — run predates per-class metrics"
    return f"```\n{format_confusion(dict(metrics))}\n```"


def _best_metrics(summary: Mapping[str, Any]) -> dict | None:
    """The metric block from the epoch that saved the model.

    Older summaries have neither `best_val_metrics` nor per-epoch `val_metrics`
    — they predate this feature — so every caller must tolerate None, the same
    way `result_embed` already falls back for `best_val_acc`.
    """
    best = summary.get("best_val_metrics")
    if best:
        return dict(best)
    best_epoch = summary.get("best_epoch")
    for epoch in summary.get("per_epoch") or []:
        if epoch.get("epoch") == best_epoch and epoch.get("val_metrics"):
            return dict(epoch["val_metrics"])
    return None


def _gb(mb: float) -> str:
    return f"{mb / 1024:.2f} GB"


def _memory_text(memory: Mapping[str, Any] | None) -> str:
    """Render whatever probes came back — see scheduler.memory_snapshot()."""
    if not memory:
        return "n/a"
    parts = []
    if memory.get("rss_peak_mb"):
        parts.append(f"peak RSS {_gb(memory['rss_peak_mb'])}")
    if memory.get("mps_allocated_mb"):
        mps = f"MPS {_gb(memory['mps_allocated_mb'])}"
        if memory.get("mps_driver_mb"):
            mps += f" ({_gb(memory['mps_driver_mb'])} driver)"
        parts.append(mps)
    if memory.get("cuda_allocated_mb"):
        parts.append(f"CUDA {_gb(memory['cuda_allocated_mb'])}")
    return " · ".join(parts) or "n/a"


def peak_memory(per_epoch: list[dict]) -> dict:
    """High-water mark across epochs, key by key."""
    peak: dict[str, float] = {}
    for epoch in per_epoch:
        for key, value in (epoch.get("memory") or {}).items():
            if isinstance(value, int | float):
                peak[key] = max(peak.get(key, 0.0), float(value))
    return peak


def checkpoint_embed(record: Mapping[str, Any]) -> dict:
    """One saved checkpoint, posted as its own message while the run continues.

    Only an improvement saves a checkpoint, so this fires ~3x in a 5-epoch run —
    a progress trail you can watch, not a per-epoch firehose.
    """
    val_loss = float(record.get("val_loss", 0.0))
    previous = record.get("previous_best_val_loss")
    delta = (
        "first checkpoint this run"
        if previous is None
        else f"▼ {float(previous) - val_loss:.4f} better than {float(previous):.4f}"
    )
    keys = record.get("checkpoint_keys") or []
    metrics = record.get("val_metrics")
    fields = [
        {"name": "Val loss", "value": f"{val_loss:.4f} — {delta}", "inline": False},
    ]
    if metrics:
        fields.append(
            {
                "name": "Macro-F1 / balanced acc",
                "value": (
                    f"{float(metrics.get('macro_f1', 0.0)):.3f} / "
                    f"{float(metrics.get('balanced_accuracy', 0.0)):.1%}"
                ),
                "inline": True,
            }
        )
    fields.extend(
        [
            {
                "name": "Val accuracy",
                "value": f"{float(record.get('val_acc', 0.0)):.1%}",
                "inline": True,
            },
            {
                "name": "Epoch time",
                "value": f"{float(record.get('duration_s', 0.0)) / 60:.1f} min",
                "inline": True,
            },
        ]
    )
    if metrics:
        # Per-class recall on the checkpoint post too: an improving val loss
        # with a collapsing Full recall is a regression wearing a green badge,
        # and this is the message that gets read while the run is still going.
        fields.append(
            {"name": "Per class", "value": _recall_text(metrics), "inline": False}
        )
    fields.extend(
        [
            {
                "name": "Memory",
                "value": _memory_text(record.get("memory")),
                "inline": False,
            },
            {"name": "Uploaded", "value": f"{len(keys)}/3 files → R2", "inline": True},
        ]
    )
    return {
        "title": f"💾 Checkpoint saved — epoch {record.get('epoch', '?')}/{record.get('total_epochs', '?')}",
        "color": COLOR_OK if len(keys) == 3 else COLOR_FAILED,
        "fields": fields,
        "footer": {"text": "is-the-mountain-out • scheduled training"},
        "timestamp": datetime.now(UTC).isoformat(),
    }


def _duration_text(duration_s: float, summary: dict) -> str:
    """Total wall time with the prefetch/epoch breakdown when the summary has it."""
    parts = [f"{duration_s / 60:.0f} min total"]
    prefetch_s = summary.get("prefetch_s")
    if prefetch_s:
        parts.append(f"prefetch {prefetch_s / 60:.1f} min")
    epoch_times = [
        e["duration_s"] for e in summary.get("per_epoch") or [] if e.get("duration_s")
    ]
    if epoch_times:
        parts.append(f"~{sum(epoch_times) / len(epoch_times) / 60:.1f} min/epoch")
    return " · ".join(parts)


def result_embed(
    summary: dict,
    pending_n: int,
    previous_best: float | None,
    duration_s: float,
    previous_best_acc: float | None = None,
    previous_macro_f1: float | None = None,
) -> dict:
    counts = summary.get("class_counts", {})
    dataset = (
        f"{summary.get('labels_loaded', '?')} labels "
        f"({counts.get('not_out', '?')}/{counts.get('full', '?')}/{counts.get('partial', '?')} "
        "Not Out/Full/Partial)"
    )
    val_counts = summary.get("val_class_counts")
    if val_counts:
        # The denominator behind every per-class rate below. Without it the
        # recall numbers look far more precise than 16 val frames can support.
        dataset += (
            f"\nval split: {val_counts.get('not_out', '?')}/"
            f"{val_counts.get('full', '?')}/{val_counts.get('partial', '?')}"
            f" (seed {summary.get('split_seed', '?')})"
        )
    best = summary.get("best_val_loss")
    best_epoch = summary.get("best_epoch")
    per_epoch = summary.get("per_epoch") or [{}]
    best_val_acc = summary.get("best_val_acc")
    if best_val_acc is None:  # older summaries: recover from the best epoch's record
        best_val_acc = next(
            (e.get("val_acc") for e in per_epoch if e.get("epoch") == best_epoch), None
        )
    last = per_epoch[-1]
    uploaded = summary.get("checkpoint_keys_uploaded") or []
    checkpoint = (
        f"{len(uploaded)}/3 files → R2 (live on next container cold start)"
        if len(uploaded) == 3
        else f"⚠️ only {len(uploaded)}/3 files uploaded — check logs"
    )
    metrics = _best_metrics(summary)
    macro_f1 = metrics.get("macro_f1") if metrics else None
    balanced = metrics.get("balanced_accuracy") if metrics else None
    return {
        "title": "🧠 Weekly training run",
        "color": COLOR_OK if len(uploaded) == 3 else COLOR_FAILED,
        # Order is the message. Macro-F1 and per-class recall come FIRST because
        # raw accuracy on an 86%-majority label set is not evidence the model
        # works — 86.3% of it is available by always answering "Not Out".
        # Accuracy stays, demoted, so run-over-run history remains readable.
        "fields": [
            {
                "name": "Trigger",
                "value": f"{pending_n} new Discord label event(s)",
                "inline": False,
            },
            {"name": "Dataset", "value": dataset, "inline": False},
            {
                "name": "Macro-F1 (headline)",
                "value": _f1_delta_text(macro_f1, previous_macro_f1),
                "inline": False,
            },
            {
                "name": "Balanced accuracy",
                "value": (
                    f"{balanced:.1%} (chance = 33.3%)"
                    if balanced is not None
                    else "n/a — run predates per-class metrics"
                ),
                "inline": True,
            },
            {
                "name": "Mountain visible (Full+Partial)",
                "value": _visible_text(metrics),
                "inline": False,
            },
            {"name": "Per class", "value": _recall_text(metrics), "inline": False},
            {
                "name": "Confusion (rows = truth)",
                "value": _confusion_text(metrics),
                "inline": False,
            },
            {
                "name": "Best val loss",
                "value": f"{best:.4f} (epoch {best_epoch}) — {_delta_text(best, previous_best)}",
                "inline": False,
            },
            {
                "name": "Val accuracy (majority baseline 86.3%)",
                "value": _acc_delta_text(best_val_acc, previous_best_acc),
                "inline": False,
            },
            {
                "name": "Final epoch",
                "value": f"val_acc {last.get('val_acc', 0.0):.1%} · train_loss {last.get('train_loss', 0.0):.4f}",
                "inline": True,
            },
            {
                "name": "Duration",
                "value": _duration_text(duration_s, summary),
                "inline": True,
            },
            {
                "name": "Peak memory",
                "value": _memory_text(peak_memory(per_epoch)),
                "inline": False,
            },
            {"name": "Checkpoint", "value": checkpoint, "inline": False},
        ],
        "footer": {"text": "is-the-mountain-out • scheduled training"},
        "timestamp": datetime.now(UTC).isoformat(),
    }


def failure_embed(pending_n: int, reason: str) -> dict:
    return {
        "title": "🧠 Weekly training run — failed",
        "color": COLOR_FAILED,
        "fields": [
            {
                "name": "Trigger",
                "value": f"{pending_n} new Discord label event(s)",
                "inline": False,
            },
            {"name": "Error", "value": reason[:1000], "inline": False},
            {
                "name": "Next",
                "value": "watermark not advanced — next weekly run retries this delta",
                "inline": False,
            },
        ],
        "footer": {"text": "is-the-mountain-out • scheduled training"},
        "timestamp": datetime.now(UTC).isoformat(),
    }


# ---------- Discord REST (best-effort; never fails the run) ----------


class Telemetry:
    def __init__(self, token: str, channel_id: str):
        self.enabled = bool(token and channel_id)
        self.token = token
        self.channel_id = channel_id
        self.message_id: str | None = None
        if not self.enabled:
            print(
                "telemetry: DISCORD_BOT_TOKEN/DISCORD_CHANNEL_ID unset — posts skipped"
            )

    def _request(self, method: str, url: str, embed: dict) -> str | None:
        try:
            resp = requests.request(
                method,
                url,
                headers={"Authorization": f"Bot {self.token}"},
                json={"embeds": [embed]},
                timeout=15,
            )
            resp.raise_for_status()
            return str(resp.json().get("id"))
        except requests.RequestException as exc:
            print(f"telemetry: Discord {method} failed (non-fatal): {exc}")
            return None

    def post(self, embed: dict) -> None:
        if not self.enabled:
            return
        self.message_id = self._request(
            "POST", f"{DISCORD_API}/channels/{self.channel_id}/messages", embed
        )

    def post_standalone(self, embed: dict) -> None:
        """Post a message that is NOT the run's main message.

        Checkpoints are their own posts, so they must not capture
        `message_id` — the final result still edits the start message.
        """
        if not self.enabled:
            return
        self._request(
            "POST", f"{DISCORD_API}/channels/{self.channel_id}/messages", embed
        )

    def update(self, embed: dict) -> None:
        if not self.enabled:
            return
        if self.message_id is None:
            self.post(embed)
            return
        self._request(
            "PATCH",
            f"{DISCORD_API}/channels/{self.channel_id}/messages/{self.message_id}",
            embed,
        )


# ---------- Live progress (tail the trainer's JSONL while it runs) ----------

PROGRESS_POLL_SECONDS = 15


def drain_progress(path: Path, offset: int, telemetry: Telemetry) -> int:
    """Post any newly-completed checkpoint lines; return the new byte offset.

    Reads bytes, not lines, and stops at the last newline: the trainer appends
    while we read, so a partial final line is left for the next poll rather
    than parsed as truncated JSON. Never raises — losing a progress post must
    not abort a training run that is otherwise fine.
    """
    try:
        if not path.exists():
            return offset
        with open(path, "rb") as handle:
            handle.seek(offset)
            chunk = handle.read()
    except OSError as exc:
        print(f"telemetry: progress read failed (non-fatal): {exc}")
        return offset

    complete = chunk.rfind(b"\n") + 1
    if complete <= 0:
        return offset
    for raw in chunk[:complete].splitlines():
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get("checkpoint_saved"):
            telemetry.post_standalone(checkpoint_embed(record))
    return offset + complete


# ---------- The run ----------


def _tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise click.ClickException(
            f"'{name}' not on PATH — run via scripts/scheduled-train.sh (uv run)."
        )
    return path


@click.command()
@click.option("--config", default="mountain.toml", help="Path to mountain.toml.")
@click.option("--data-root", default="data", help="Local data root.")
@click.option(
    "--check", is_flag=True, help="Report pending events and exit (no side effects)."
)
def main(config: str, data_root: str, check: bool) -> None:
    """Gated weekly retrain: skip when idle, train + report when labels arrived."""
    import os

    loader = ConfigLoader(config)
    storage = uncached_storage(loader.get_storage(data_root))
    epochs = int(loader.data.get("training", {}).get("scheduled_epochs", 5))

    try:
        events_text = storage.get_text(EVENTS_KEY)
    except Exception:  # noqa: BLE001 — no events log yet means nothing to train on
        events_text = ""
    watermark = load_watermark(storage)
    pending = pending_events(parse_events(events_text), watermark.get("last_event_ts"))

    if check:
        print(
            f"pending events: {len(pending)} "
            f"(watermark: {watermark.get('last_event_ts', 'none')})"
        )
        return
    if not pending:
        print(
            "no new label events since last run — skipping (this is the normal idle path)"
        )
        return

    started_at = datetime.now(UTC)
    newest_ts = max(e["timestamp"] for e in pending)
    telemetry = Telemetry(
        os.environ.get("DISCORD_BOT_TOKEN", ""),
        os.environ.get("DISCORD_CHANNEL_ID", ""),
    )
    print(f"{len(pending)} new label event(s) — starting {epochs}-epoch batch run")
    telemetry.post(start_embed(len(pending), epochs, started_at))

    pull = subprocess.run(
        [_tool("collect"), "sync", "labels", "pull", "--config", config], check=False
    )
    if pull.returncode != 0:
        telemetry.update(failure_embed(len(pending), "labels pull from R2 failed"))
        raise click.ClickException("labels pull failed; aborting before training")

    run_dir = Path(tempfile.mkdtemp(prefix="mountain-train-"))
    summary_path = run_dir / "summary.json"
    progress_path = run_dir / "progress.jsonl"

    # Popen, not run(): the trainer takes ~an hour, and the point of the
    # progress file is to report checkpoints WHILE it works rather than
    # reconstructing them from the summary afterwards.
    proc = subprocess.Popen(
        [
            _tool("training"),
            "batch",
            "--labels",
            str(Path(data_root) / "labels.yaml"),
            "--epochs",
            str(epochs),
            "--config",
            config,
            "--json-summary",
            str(summary_path),
            "--progress-jsonl",
            str(progress_path),
        ],
    )
    offset = 0
    while proc.poll() is None:
        time.sleep(PROGRESS_POLL_SECONDS)
        offset = drain_progress(progress_path, offset, telemetry)
    # Final drain: the last epoch's checkpoint is usually written between the
    # previous poll and process exit.
    drain_progress(progress_path, offset, telemetry)

    returncode = proc.returncode
    duration_s = (datetime.now(UTC) - started_at).total_seconds()

    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except ValueError:
            summary = {}

    if returncode != 0 or summary.get("status") != "ok":
        reason = f"training exit {returncode}, summary status: {summary.get('status', 'missing')}"
        print(f"run failed — {reason}; watermark NOT advanced")
        telemetry.update(failure_embed(len(pending), reason))
        sys.exit(1)

    best_metrics = _best_metrics(summary) or {}
    save_watermark(
        storage,
        {
            "last_event_ts": newest_ts,
            "ran_at": started_at.isoformat(),
            "labels_total": summary.get("labels_loaded"),
            "best_val_loss": summary.get("best_val_loss"),
            "best_val_acc": summary.get("best_val_acc"),
            # Carried so NEXT week's embed can show a macro-F1 delta. Absent on
            # watermarks written before this landed — every reader defaults it.
            "best_macro_f1": best_metrics.get("macro_f1"),
            "best_balanced_accuracy": best_metrics.get("balanced_accuracy"),
            "epochs": epochs,
        },
    )
    telemetry.update(
        result_embed(
            summary,
            len(pending),
            watermark.get("best_val_loss"),
            duration_s,
            previous_best_acc=watermark.get("best_val_acc"),
            previous_macro_f1=watermark.get("best_macro_f1"),
        )
    )
    print(
        f"run complete — best val_loss {summary.get('best_val_loss'):.4f}, "
        f"watermark → {newest_ts}"
    )


if __name__ == "__main__":
    main()

"""Reaction-labeling logic for the Discord bot — pure functions, no discord.py.

Everything here runs and tests without a Discord connection or the optional
`bot` dependency group installed; bot/main.py wires these into discord.py
events. The label store contract is identical to tools/classifier_server.py:
labels.yaml on the configured storage backend is the source of truth, updated
with a union merge that never deletes keys added elsewhere.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from datetime import time as dtime
from typing import Any
from zoneinfo import ZoneInfo

import requests
import yaml

logger = logging.getLogger(__name__)

# Mirrors tools/predict_state.py CLASS_NAMES and the labels.yaml convention:
# 0 = Not Out, 1 = Full, 2 = Partial.
CLASS_NAMES = ["not_out", "full", "partial"]
CLASS_LABELS = ["Not Out", "Full", "Partial"]

# The reaction vocabulary. Skin-tone and variation-selector variants normalize
# to these base emoji before lookup (a 👍🏻 tap counts as 👍).
EMOJI_TO_CLASS = {
    "👍": 1,  # Full
    "⛅": 2,  # Partial
    "👎": 0,  # Not Out
}

LABELS_KEY = "labels.yaml"  # same object the classifier UI and trainer share
EVENTS_KEY = "labels/discord-events.jsonl"  # append-only provenance log

# Embed colors — mirror worker/src/discord-mountain-notify.ts.
COLOR_VISIBLE = 0x2ECC71
COLOR_NOT_VISIBLE = 0x95A5A6
COLOR_NEUTRAL = 0x3498DB

_SKIN_TONES = {chr(cp) for cp in range(0x1F3FB, 0x1F400)}
_VARIATION_SELECTOR = "\ufe0f"

# A capture key as produced by collect.collector.perform_capture:
#   YYYYMMDD/HHMMSS_ffffff_UTC/images/<name>.jpg
_CAPTURE_KEY_RE = re.compile(r"^\d{8}/\d{6}_\d{6}_UTC/images/[^/]+\.jpg$")


# ---------- Emoji → label ----------


def normalize_emoji(emoji: str) -> str:
    """Strip skin-tone modifiers and variation selectors: 👍🏻 → 👍."""
    return "".join(
        ch for ch in emoji if ch not in _SKIN_TONES and ch != _VARIATION_SELECTOR
    )


def class_for_emoji(emoji: str) -> int | None:
    """Label class for a reaction emoji, or None if it isn't a label emoji."""
    return EMOJI_TO_CLASS.get(normalize_emoji(emoji))


def reaction_legend() -> str:
    """Human-readable reaction → class legend, ordered by class index."""
    by_class = sorted(EMOJI_TO_CLASS.items(), key=lambda kv: kv[1], reverse=True)
    return " · ".join(f"{emoji} {CLASS_LABELS[cls]}" for emoji, cls in by_class)


# ---------- Message ↔ capture mapping ----------
#
# The embed footer carries the capture's storage key verbatim — on the bot's
# own hourly posts AND on the Worker's transition notifications, which persist
# the frame they announce and footer its key (worker/src/index.ts). It is the
# only message↔capture state, so labeling survives restarts with nothing on
# disk. Prose footers (historical notifications, anything else) never parse as
# a capture, and the wiring additionally gates parsed keys on existing in
# storage before recording a label.


def footer_for_capture(capture_key: str) -> str:
    return capture_key


def capture_key_from_footer(text: str | None) -> str | None:
    if not text:
        return None
    text = text.strip()
    return text if _CAPTURE_KEY_RE.match(text) else None


# ---------- Settings ----------


@dataclass(frozen=True)
class BotSettings:
    token: str
    channel_id: int
    allowed_user_ids: frozenset[int]  # empty → any human reaction counts
    post_interval_seconds: int = 3600
    sweep_hours: int = 72
    state_url: str = ""
    window_start: dtime | None = None  # fixed window; overrides solar
    window_end: dtime | None = None
    timezone: str = "America/Los_Angeles"
    latitude: float = 0.0
    longitude: float = 0.0

    @classmethod
    def from_mapping(
        cls, config: Mapping[str, Any], env: Mapping[str, str] | None = None
    ) -> BotSettings:
        """Build settings from the mountain.toml data dict + environment.

        Secrets and per-deployment ids come from the environment (cf.env);
        behavior tuning comes from the optional [bot] section. Raises
        ValueError with a setup hint when required env keys are missing.
        """
        env = os.environ if env is None else env
        bot_cfg = config.get("bot", {})
        webcam = config.get("webcam", {})

        token = env.get("DISCORD_BOT_TOKEN", "").strip()
        channel_raw = env.get("DISCORD_CHANNEL_ID", "").strip()
        if not token or not channel_raw:
            raise ValueError(
                "DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID must be set "
                "(source cf.env — see BOT.md for setup)."
            )

        allowed_raw = env.get("DISCORD_ALLOWED_USER_IDS", "")
        allowed = frozenset(
            int(part) for part in re.split(r"[,\s]+", allowed_raw) if part
        )

        def parse_window(key: str) -> dtime | None:
            raw = bot_cfg.get(key)
            return dtime.fromisoformat(raw) if raw else None

        return cls(
            token=token,
            channel_id=int(channel_raw),
            allowed_user_ids=allowed,
            post_interval_seconds=int(bot_cfg.get("post_interval_seconds", 3600)),
            sweep_hours=int(bot_cfg.get("sweep_hours", 72)),
            state_url=str(bot_cfg.get("state_url", "")),
            window_start=parse_window("window_start"),
            window_end=parse_window("window_end"),
            timezone=str(bot_cfg.get("timezone", "America/Los_Angeles")),
            latitude=float(webcam.get("latitude", 0.0)),
            longitude=float(webcam.get("longitude", 0.0)),
        )


# ---------- Posting schedule ----------


def next_post_due(
    last_post: datetime | None, now: datetime, interval_seconds: int
) -> bool:
    return last_post is None or (now - last_post).total_seconds() >= interval_seconds


def in_daylight_window(settings: BotSettings, now: datetime) -> bool:
    """True when *now* falls inside the posting window.

    A fixed [bot] window_start/window_end pair wins when configured; otherwise
    the window is civil dawn→dusk at the webcam's coordinates (astral — the
    same library behind `collect schedule`'s solar-aligned capture plans).
    """
    local = now.astimezone(ZoneInfo(settings.timezone))
    if settings.window_start is not None and settings.window_end is not None:
        return settings.window_start <= local.time() <= settings.window_end

    from astral import Observer
    from astral.sun import sun

    observer = Observer(latitude=settings.latitude, longitude=settings.longitude)
    times = sun(observer, date=local.date(), tzinfo=local.tzinfo)
    return times["dawn"] <= local <= times["dusk"]


# ---------- Prediction context (public state.json) ----------


def fetch_prediction(state_url: str, timeout: float = 10.0) -> dict | None:
    """Fetch the Worker-published PredictionState, or None (never raises)."""
    if not state_url:
        return None
    try:
        resp = requests.get(state_url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning(f"state.json fetch failed (non-fatal): {exc}")
        return None


# ---------- Embed ----------


def build_capture_embed(
    capture_key: str,
    captured_at: datetime,
    prediction: Mapping[str, Any] | None = None,
) -> dict:
    """Embed dict (discord.Embed.from_dict shape) for a labelable capture post.

    Styling mirrors worker/src/discord-mountain-notify.ts; the image is the
    attached capture, and the footer is the capture key.
    """
    fields = [
        {"name": "Label with a reaction", "value": reaction_legend(), "inline": False}
    ]
    color = COLOR_NEUTRAL
    if prediction:
        name = str(prediction.get("class_name", "?"))
        pretty = CLASS_LABELS[CLASS_NAMES.index(name)] if name in CLASS_NAMES else name
        confidence = prediction.get("confidence") or {}
        try:
            pct = f"{float(confidence.get(name, 0.0)) * 100:.1f}%"
        except TypeError, ValueError:
            pct = "?"
        fields.insert(
            0, {"name": "Model says", "value": f"{pretty} ({pct})", "inline": True}
        )
        color = COLOR_VISIBLE if prediction.get("is_out") else COLOR_NOT_VISIBLE

    return {
        "title": "🏔️ Is the mountain out?",
        "color": color,
        "fields": fields,
        "image": {"url": "attachment://capture.jpg"},
        "footer": {"text": footer_for_capture(capture_key)},
        "timestamp": captured_at.isoformat(),
    }


# ---------- Label recording ----------


def load_labels(storage, labels_key: str = LABELS_KEY) -> dict:
    """Current labels from storage; empty dict when absent/unreadable."""
    try:
        return yaml.safe_load(storage.get_text(labels_key)) or {}
    except Exception:  # noqa: BLE001 — backend-specific miss/IO errors all mean "no labels yet"
        return {}


def record_label(
    storage,
    capture_key: str,
    label: int,
    *,
    user_id: int,
    source: str = "discord-reaction",
    labels_key: str = LABELS_KEY,
    events_key: str = EVENTS_KEY,
    now: datetime | None = None,
) -> dict:
    """Merge {capture_key: label} into labels.yaml and append a provenance event.

    Same union-merge contract as tools/classifier_server.py save_labels():
    fetch the shared file, update, put back — never delete keys added
    elsewhere. Re-labeling a capture overwrites its entry (last reaction wins).
    """
    labels = load_labels(storage, labels_key)
    labels[capture_key] = int(label)
    storage.put_text(labels_key, yaml.safe_dump(labels))

    event = {
        "timestamp": (now or datetime.now(UTC)).isoformat(),
        "capture": capture_key,
        "label": int(label),
        "class": CLASS_NAMES[int(label)],
        "user_id": user_id,
        "source": source,
    }
    try:
        existing = storage.get_text(events_key)
    except Exception:  # noqa: BLE001 — backend-specific miss/IO errors all mean "no log yet"
        existing = ""
    storage.put_text(events_key, existing + json.dumps(event) + "\n")
    return labels


# ---------- Reaction resolution ----------


def labeler_allowed(user_id: int, is_bot: bool, allowed: frozenset[int]) -> bool:
    """Whether a reactor's labels count. Empty allowlist = any human."""
    if is_bot:
        return False
    return not allowed or user_id in allowed


def label_from_reaction_users(
    reaction_users: Mapping[str, Iterable[int]], allowed: frozenset[int]
) -> int | None:
    """Resolve a message's accumulated reactions to one label (startup sweep).

    *reaction_users* maps emoji → non-bot user ids currently reacted. Majority
    of allowlisted votes wins; a tie between different classes is ambiguous →
    None. (The live handler is last-reaction-wins; this path only catches up
    on reactions added while the bot was offline.)
    """
    votes: dict[int, int] = {}
    for emoji, users in reaction_users.items():
        cls = class_for_emoji(emoji)
        if cls is None:
            continue
        count = sum(1 for uid in users if not allowed or uid in allowed)
        if count:
            votes[cls] = votes.get(cls, 0) + count
    if not votes:
        return None
    best = max(votes.values())
    top = [cls for cls, count in votes.items() if count == best]
    return top[0] if len(top) == 1 else None

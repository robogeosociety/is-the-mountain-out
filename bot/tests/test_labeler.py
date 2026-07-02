"""Tests for bot.labeler — pure logic, no Discord connection or discord.py.

Storage-facing tests run against the real LocalStorage backend in a tmp dir,
mirroring collect/tests/test_storage.py.
"""

import json
from datetime import UTC, datetime, time as dtime
from unittest.mock import patch

import pytest
import requests
import yaml

from bot import labeler
from bot.labeler import BotSettings
from collect.storage import LocalStorage


# ---------- Emoji normalization + mapping ----------


class TestEmoji:
    @pytest.mark.parametrize(
        "variant,base",
        [
            ("👍🏻", "👍"),
            ("👍🏿", "👍"),
            ("👎🏽", "👎"),
            ("👍", "👍"),
            ("⛅", "⛅"),
            ("☁️", "☁"),  # variation selector stripped
        ],
    )
    def test_normalize(self, variant, base):
        assert labeler.normalize_emoji(variant) == base

    @pytest.mark.parametrize(
        "emoji,cls",
        [("👍", 1), ("⛅", 2), ("👎", 0), ("👍🏻", 1), ("👎🏿", 0)],
    )
    def test_label_emoji(self, emoji, cls):
        assert labeler.class_for_emoji(emoji) == cls

    @pytest.mark.parametrize("emoji", ["❤️", "🏔️", "x", ""])
    def test_non_label_emoji(self, emoji):
        assert labeler.class_for_emoji(emoji) is None

    def test_legend_covers_all_classes(self):
        legend = labeler.reaction_legend()
        for label in labeler.CLASS_LABELS:
            assert label in legend


# ---------- Footer round-trip ----------


class TestFooter:
    KEY = "20260701/143000_123456_UTC/images/143000_123456_UTC_cam.jpg"

    def test_round_trip(self):
        assert (
            labeler.capture_key_from_footer(labeler.footer_for_capture(self.KEY))
            == self.KEY
        )

    def test_strips_whitespace(self):
        assert labeler.capture_key_from_footer(f"  {self.KEY} ") == self.KEY

    @pytest.mark.parametrize(
        "text",
        [
            None,
            "",
            "is-the-mountain-out • Automated prediction",  # Worker webhook footer
            "20260701/143000_UTC/images/x.jpg",  # missing microseconds
            "20260701/143000_123456_UTC/metar/metar.txt",  # not an image
            "../../etc/passwd",
        ],
    )
    def test_rejects_foreign_footers(self, text):
        assert labeler.capture_key_from_footer(text) is None


# ---------- Settings ----------


BASE_CONFIG = {
    "bot": {
        "post_interval_seconds": 1800,
        "state_url": "https://example.com/state.json",
    },
    "webcam": {"latitude": 47.6533, "longitude": -122.3091},
}


class TestBotSettings:
    def test_from_mapping(self):
        env = {
            "DISCORD_BOT_TOKEN": "tok",
            "DISCORD_CHANNEL_ID": "1234",
            "DISCORD_ALLOWED_USER_IDS": "111, 222",
        }
        s = BotSettings.from_mapping(BASE_CONFIG, env)
        assert s.token == "tok"
        assert s.channel_id == 1234
        assert s.allowed_user_ids == frozenset({111, 222})
        assert s.post_interval_seconds == 1800
        assert s.state_url == "https://example.com/state.json"
        assert s.latitude == pytest.approx(47.6533)
        assert s.sweep_hours == 72  # default

    def test_missing_token_raises(self):
        with pytest.raises(ValueError, match="DISCORD_BOT_TOKEN"):
            BotSettings.from_mapping(BASE_CONFIG, {"DISCORD_CHANNEL_ID": "1"})

    def test_missing_channel_raises(self):
        with pytest.raises(ValueError, match="DISCORD_CHANNEL_ID"):
            BotSettings.from_mapping(BASE_CONFIG, {"DISCORD_BOT_TOKEN": "tok"})

    def test_empty_allowlist_and_bot_section_optional(self):
        env = {"DISCORD_BOT_TOKEN": "tok", "DISCORD_CHANNEL_ID": "9"}
        s = BotSettings.from_mapping({"webcam": {}}, env)
        assert s.allowed_user_ids == frozenset()
        assert s.post_interval_seconds == 3600

    def test_fixed_window_parsed(self):
        cfg = {"bot": {"window_start": "06:00", "window_end": "21:00"}, "webcam": {}}
        env = {"DISCORD_BOT_TOKEN": "t", "DISCORD_CHANNEL_ID": "1"}
        s = BotSettings.from_mapping(cfg, env)
        assert s.window_start == dtime(6, 0)
        assert s.window_end == dtime(21, 0)


def _settings(**overrides) -> BotSettings:
    base = dict(
        token="t",
        channel_id=1,
        allowed_user_ids=frozenset(),
        latitude=47.6533,
        longitude=-122.3091,
        timezone="America/Los_Angeles",
    )
    base.update(overrides)
    return BotSettings(**base)


# ---------- Posting schedule ----------


class TestSchedule:
    def test_due_when_never_posted(self):
        assert labeler.next_post_due(None, datetime.now(UTC), 3600)

    def test_not_due_within_interval(self):
        now = datetime(2026, 7, 1, 12, 30, tzinfo=UTC)
        last = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
        assert not labeler.next_post_due(last, now, 3600)

    def test_due_after_interval(self):
        now = datetime(2026, 7, 1, 13, 0, tzinfo=UTC)
        last = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
        assert labeler.next_post_due(last, now, 3600)

    def test_fixed_window(self):
        s = _settings(window_start=dtime(6, 0), window_end=dtime(21, 0))
        noon_pt = datetime(2026, 7, 1, 19, 0, tzinfo=UTC)  # 12:00 PDT
        midnight_pt = datetime(2026, 7, 1, 7, 30, tzinfo=UTC)  # 00:30 PDT
        assert labeler.in_daylight_window(s, noon_pt)
        assert not labeler.in_daylight_window(s, midnight_pt)

    def test_solar_window_seattle(self):
        # Real astral computation at the webcam's coordinates: Seattle summer
        # noon is inside civil dawn→dusk, 1 AM is not.
        s = _settings()
        noon_pt = datetime(2026, 7, 1, 19, 0, tzinfo=UTC)
        one_am_pt = datetime(2026, 7, 1, 8, 0, tzinfo=UTC)
        assert labeler.in_daylight_window(s, noon_pt)
        assert not labeler.in_daylight_window(s, one_am_pt)


# ---------- Prediction fetch ----------


class TestFetchPrediction:
    def test_empty_url_returns_none(self):
        assert labeler.fetch_prediction("") is None

    def test_network_error_returns_none(self):
        with patch("bot.labeler.requests.get", side_effect=requests.ConnectionError):
            assert labeler.fetch_prediction("https://example.com/state.json") is None

    def test_ok_returns_json(self):
        with patch("bot.labeler.requests.get") as mock_get:
            mock_get.return_value.json.return_value = {"is_out": True}
            mock_get.return_value.raise_for_status.return_value = None
            assert labeler.fetch_prediction("https://x/state.json") == {"is_out": True}


# ---------- Embed ----------


class TestEmbed:
    KEY = "20260701/143000_123456_UTC/images/cam.jpg"
    TS = datetime(2026, 7, 1, 14, 30, tzinfo=UTC)

    def test_footer_is_capture_key(self):
        embed = labeler.build_capture_embed(self.KEY, self.TS)
        assert embed["footer"]["text"] == self.KEY
        assert labeler.capture_key_from_footer(embed["footer"]["text"]) == self.KEY

    def test_without_prediction(self):
        embed = labeler.build_capture_embed(self.KEY, self.TS, None)
        assert embed["color"] == labeler.COLOR_NEUTRAL
        assert len(embed["fields"]) == 1
        assert embed["image"]["url"] == "attachment://capture.jpg"

    def test_with_prediction(self):
        prediction = {
            "class_name": "full",
            "is_out": True,
            "confidence": {"not_out": 0.05, "full": 0.873, "partial": 0.077},
        }
        embed = labeler.build_capture_embed(self.KEY, self.TS, prediction)
        assert embed["color"] == labeler.COLOR_VISIBLE
        model_field = embed["fields"][0]
        assert model_field["name"] == "Model says"
        assert model_field["value"] == "Full (87.3%)"

    def test_with_not_out_prediction(self):
        prediction = {
            "class_name": "not_out",
            "is_out": False,
            "confidence": {"not_out": 0.99},
        }
        embed = labeler.build_capture_embed(self.KEY, self.TS, prediction)
        assert embed["color"] == labeler.COLOR_NOT_VISIBLE
        assert "Not Out" in embed["fields"][0]["value"]


# ---------- Label recording ----------


class TestRecordLabel:
    KEY = "20260701/143000_123456_UTC/images/cam.jpg"

    def test_creates_labels_file(self, tmp_path):
        store = LocalStorage(str(tmp_path))
        merged = labeler.record_label(store, self.KEY, 1, user_id=111)
        assert merged == {self.KEY: 1}
        assert yaml.safe_load(store.get_text("labels.yaml")) == {self.KEY: 1}

    def test_union_merge_preserves_other_keys(self, tmp_path):
        store = LocalStorage(str(tmp_path))
        store.put_text("labels.yaml", yaml.safe_dump({"other/images/a.jpg": 0}))
        labeler.record_label(store, self.KEY, 2, user_id=111)
        merged = yaml.safe_load(store.get_text("labels.yaml"))
        assert merged == {"other/images/a.jpg": 0, self.KEY: 2}

    def test_relabel_overwrites(self, tmp_path):
        store = LocalStorage(str(tmp_path))
        labeler.record_label(store, self.KEY, 1, user_id=111)
        labeler.record_label(store, self.KEY, 0, user_id=111)
        assert yaml.safe_load(store.get_text("labels.yaml")) == {self.KEY: 0}

    def test_events_appended(self, tmp_path):
        store = LocalStorage(str(tmp_path))
        ts = datetime(2026, 7, 1, 14, 30, tzinfo=UTC)
        labeler.record_label(store, self.KEY, 1, user_id=111, now=ts)
        labeler.record_label(
            store, self.KEY, 0, user_id=222, now=ts, source="discord-sweep"
        )
        lines = store.get_text(labeler.EVENTS_KEY).strip().splitlines()
        assert len(lines) == 2
        first, second = (json.loads(line) for line in lines)
        assert first == {
            "timestamp": ts.isoformat(),
            "capture": self.KEY,
            "label": 1,
            "class": "full",
            "user_id": 111,
            "source": "discord-reaction",
        }
        assert second["source"] == "discord-sweep"

    def test_load_labels_missing_file(self, tmp_path):
        assert labeler.load_labels(LocalStorage(str(tmp_path))) == {}


# ---------- Reaction resolution ----------


class TestReactionResolution:
    def test_allowlist_empty_allows_humans(self):
        assert labeler.labeler_allowed(42, is_bot=False, allowed=frozenset())

    def test_allowlist_blocks_bots(self):
        assert not labeler.labeler_allowed(42, is_bot=True, allowed=frozenset())

    def test_allowlist_enforced(self):
        allowed = frozenset({111})
        assert labeler.labeler_allowed(111, is_bot=False, allowed=allowed)
        assert not labeler.labeler_allowed(222, is_bot=False, allowed=allowed)

    def test_sweep_single_reaction(self):
        assert labeler.label_from_reaction_users({"👍": [111]}, frozenset()) == 1

    def test_sweep_skin_tone_variant(self):
        assert labeler.label_from_reaction_users({"👎🏻": [111]}, frozenset()) == 0

    def test_sweep_majority_wins(self):
        users = {"👍": [1, 2], "👎": [3]}
        assert labeler.label_from_reaction_users(users, frozenset()) == 1

    def test_sweep_tie_is_ambiguous(self):
        users = {"👍": [1], "👎": [2]}
        assert labeler.label_from_reaction_users(users, frozenset()) is None

    def test_sweep_ignores_non_label_emoji(self):
        assert labeler.label_from_reaction_users({"❤️": [1, 2, 3]}, frozenset()) is None

    def test_sweep_respects_allowlist(self):
        users = {"👍": [999], "👎": [111]}
        assert labeler.label_from_reaction_users(users, frozenset({111})) == 0

    def test_sweep_empty(self):
        assert labeler.label_from_reaction_users({}, frozenset()) is None

"""Is the webcam still sending new pictures?

A broadcast still camera fails quietly: the URL keeps returning 200 with the
same JPEG for hours. The classifier is perfectly happy to keep predicting on a
frozen frame, which is worse than saying nothing — the site would report "the
mountain is out" long after the sun set on the frame that said so.

So each tick hashes the bytes it fetched and compares them with the last tick's
hash. The KING 5 feed refreshes about every four minutes and the tick runs
every fifteen, so two consecutive ticks with identical bytes is already odd and
three is a dead feed. At that point inference is skipped entirely and
`state.json` carries `status: "stale"` instead of a prediction.

Pure logic, no I/O — the caller owns the JSON file.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

#: Consecutive identical fetches before the feed is declared stale.
DEFAULT_STALE_AFTER_REPEATS = 3


@dataclass(frozen=True)
class FeedState:
    """Bookkeeping carried between ticks (persisted as JSON)."""

    #: sha256 of the last frame's bytes.
    last_sha256: Optional[str] = None
    #: How many consecutive fetches have returned exactly these bytes (>= 1).
    repeat_count: int = 0
    #: When these bytes were first seen (ISO 8601).
    first_seen: Optional[str] = None
    #: When the feed last delivered something new (ISO 8601).
    last_changed: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "FeedState":
        data = data or {}
        return cls(
            last_sha256=data.get("last_sha256"),
            repeat_count=int(data.get("repeat_count") or 0),
            first_seen=data.get("first_seen"),
            last_changed=data.get("last_changed"),
        )

    def to_dict(self) -> dict:
        return {
            "last_sha256": self.last_sha256,
            "repeat_count": self.repeat_count,
            "first_seen": self.first_seen,
            "last_changed": self.last_changed,
        }


def update_feed_state(
    state: FeedState,
    sha256: str,
    now: str,
    stale_after_repeats: int = DEFAULT_STALE_AFTER_REPEATS,
) -> tuple[FeedState, bool]:
    """Fold this tick's frame hash in. Returns ``(next_state, is_stale)``.

    *is_stale* is True once the same bytes have come back
    *stale_after_repeats* times in a row, and stays True until they change —
    a stale feed should not flap back to "fine" on the fourth identical fetch.
    """
    if sha256 == state.last_sha256:
        nxt = replace(state, repeat_count=state.repeat_count + 1)
    else:
        # New bytes: the feed is alive, whatever it was doing before.
        nxt = FeedState(
            last_sha256=sha256,
            repeat_count=1,
            first_seen=now,
            last_changed=now,
        )

    return nxt, nxt.repeat_count >= max(1, stale_after_repeats)

"""What the channel says, and when — a pure state machine, no I/O.

Ported from the retired Cloudflare Worker's `worker/src/transition.ts` when
inference moved onto the Mac mini (2026-09). The rules are unchanged; only the
caller is new: `tools/predict_state.py` runs this on each 15-minute tick and
*queues* the decision to `announce.jsonl` instead of posting to Discord itself.
The single Discord bot drains that queue.

The channel does two different jobs, and conflating them made it bad at both:

  ALERT         "the mountain is out" — must be TRUSTWORTHY. You act on it.
  LABEL REQUEST "I'm not sure — which is it?" — must be INFORMATIVE. You
                correct it, and the correction is worth a training row.

They want opposite inputs. An alert is only worth posting when the model is
confident; a label is only worth asking for when it ISN'T. Announcing every
confirmed change sent the *least* reliable predictions out as alerts.

So confidence routes the message. Two gates, in order:

  1. DEBOUNCE (duration). A change must hold for two consecutive ticks.
     Predictions flap — out 16:46, gone 17:01, out 18:01 is one change, not
     three.
  2. CONFIDENCE (certainty). A held change only ALERTS above
     `alert_min_confidence`. Below it, `pending` stays armed: the alert is
     delayed until the model is sure, never dropped.

Confidence here is BINARY (p(out) = full + partial, vs p(not out)), because
that is the question being asked.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Optional

#: Binary confidence at or above which a held change is announced.
DEFAULT_ALERT_MIN_CONFIDENCE = 0.85
#: Minimum gap between label requests, in seconds.
DEFAULT_LABEL_COOLDOWN_SECONDS = 4 * 60 * 60


@dataclass(frozen=True)
class NotifyState:
    """Bookkeeping carried from one tick to the next (persisted as JSON)."""

    #: Visibility the channel was last told about. None = never announced.
    announced_is_out: Optional[bool] = None
    #: A disagreeing observation awaiting confirmation on the next tick.
    pending_is_out: Optional[bool] = None
    #: When `pending_is_out` was armed (ISO 8601), for debugging.
    pending_since: Optional[str] = None
    #: When the last label request went out (ISO 8601) — drives the cooldown.
    last_label_request: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "NotifyState":
        data = data or {}
        return cls(
            announced_is_out=data.get("announced_is_out"),
            pending_is_out=data.get("pending_is_out"),
            pending_since=data.get("pending_since"),
            last_label_request=data.get("last_label_request"),
        )

    def to_dict(self) -> dict:
        return {
            "announced_is_out": self.announced_is_out,
            "pending_is_out": self.pending_is_out,
            "pending_since": self.pending_since,
            "last_label_request": self.last_label_request,
        }


@dataclass(frozen=True)
class TransitionDecision:
    """`alert` announces a change; `label` asks which it is; `quiet` posts nothing."""

    kind: str
    next: NotifyState = field(default_factory=NotifyState)


def _parse(iso: Optional[str]) -> Optional[datetime]:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None


def decide_transition(
    state: NotifyState,
    is_out: bool,
    confidence: float,
    now: str,
    alert_min_confidence: float = DEFAULT_ALERT_MIN_CONFIDENCE,
    label_cooldown_seconds: float = DEFAULT_LABEL_COOLDOWN_SECONDS,
) -> TransitionDecision:
    """Route this tick to an alert, a label request, or silence.

    :param state: persisted state from the previous tick
    :param is_out: this tick's prediction
    :param confidence: BINARY confidence in *is_out* (p(out) or p(not out))
    :param now: this tick's timestamp, ISO 8601
    """
    confident = confidence >= alert_min_confidence

    def label_due() -> bool:
        # A label request is due whenever the model is unsure and the cooldown
        # has elapsed. Evaluated last, so an alert always wins the tick.
        if confident:
            return False
        last = _parse(state.last_label_request)
        now_dt = _parse(now)
        if last is None or now_dt is None:
            return True
        return (now_dt - last).total_seconds() >= label_cooldown_seconds

    def as_label(nxt: NotifyState) -> TransitionDecision:
        return TransitionDecision("label", replace(nxt, last_label_request=now))

    # First observation ever (or after the state file is lost): adopt the
    # current visibility silently. Announcing here would fire on every restart
    # that resets the file, which is noise, not news.
    if state.announced_is_out is None:
        nxt = replace(
            state, announced_is_out=is_out, pending_is_out=None, pending_since=None
        )
        return as_label(nxt) if label_due() else TransitionDecision("quiet", nxt)

    # Agrees with what the channel already knows — nothing to announce, and any
    # half-armed flip is now disproven.
    if is_out == state.announced_is_out:
        nxt = replace(state, pending_is_out=None, pending_since=None)
        return as_label(nxt) if label_due() else TransitionDecision("quiet", nxt)

    # Disagrees and held for a second consecutive tick. Announce it only if the
    # model is sure; otherwise keep pending armed so the alert is DELAYED to the
    # first confident tick rather than dropped — and meanwhile the frame is
    # exactly the kind worth asking about.
    if state.pending_is_out == is_out and confident:
        return TransitionDecision(
            "alert",
            replace(
                state, announced_is_out=is_out, pending_is_out=None, pending_since=None
            ),
        )

    # First tick of a disagreement, or a held-but-unconfident one: (re)arm.
    nxt = replace(
        state,
        pending_is_out=is_out,
        pending_since=state.pending_since if state.pending_is_out == is_out else now,
    )
    return as_label(nxt) if label_due() else TransitionDecision("quiet", nxt)


def binary_confidence(confidence: dict, is_out: bool) -> float:
    """p(out) = full + partial when *is_out*, else p(not out).

    The three-class head answers a question nobody asks. The channel's question
    is binary, so the gate has to be too.
    """
    if is_out:
        return float(confidence.get("full", 0.0)) + float(
            confidence.get("partial", 0.0)
        )
    return float(confidence.get("not_out", 0.0))

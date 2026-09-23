"""Ported from the retired Worker's worker/test/transition.test.ts."""

from bot.transition import (
    NotifyState,
    binary_confidence,
    decide_transition,
)

T0 = "2026-09-20T00:00:00Z"
T1 = "2026-09-20T00:15:00Z"
T2 = "2026-09-20T00:30:00Z"
LATER = "2026-09-20T06:00:00Z"


def test_first_observation_is_adopted_silently():
    d = decide_transition(NotifyState(), True, 0.99, T0)
    assert d.kind == "quiet"
    assert d.next.announced_is_out is True


def test_agreeing_tick_is_quiet_and_disarms_pending():
    state = NotifyState(announced_is_out=False, pending_is_out=True, pending_since=T0)
    d = decide_transition(state, False, 0.99, T1)
    assert d.kind == "quiet"
    assert d.next.pending_is_out is None
    assert d.next.pending_since is None


def test_change_needs_two_consecutive_ticks():
    state = NotifyState(announced_is_out=False)
    first = decide_transition(state, True, 0.99, T0)
    assert first.kind == "quiet"
    assert first.next.pending_is_out is True
    assert first.next.announced_is_out is False

    second = decide_transition(first.next, True, 0.99, T1)
    assert second.kind == "alert"
    assert second.next.announced_is_out is True
    assert second.next.pending_is_out is None


def test_held_but_unconfident_change_is_delayed_not_dropped():
    # last_label_request set so the label cooldown keeps this tick quiet and the
    # assertion is about the alert being *held*, not about the label request.
    state = NotifyState(
        announced_is_out=False,
        pending_is_out=True,
        pending_since=T0,
        last_label_request=T0,
    )
    unsure = decide_transition(state, True, 0.60, T1)
    assert unsure.kind == "quiet"
    assert unsure.next.pending_is_out is True
    # pending_since is preserved across a held-but-unconfident tick
    assert unsure.next.pending_since == T0

    confident = decide_transition(unsure.next, True, 0.95, T2)
    assert confident.kind == "alert"


def test_unsure_tick_asks_for_a_label_once_per_cooldown():
    state = NotifyState(announced_is_out=False)
    first = decide_transition(state, False, 0.40, T0)
    assert first.kind == "label"
    assert first.next.last_label_request == T0

    soon = decide_transition(first.next, False, 0.40, T1)
    assert soon.kind == "quiet"

    after = decide_transition(first.next, False, 0.40, LATER)
    assert after.kind == "label"


def test_alert_beats_a_due_label_request():
    # Confident ticks never ask for labels, so a confirmed change alerts.
    state = NotifyState(announced_is_out=False, pending_is_out=True, pending_since=T0)
    d = decide_transition(state, True, 0.99, T1)
    assert d.kind == "alert"
    assert d.next.last_label_request is None


def test_binary_confidence_collapses_the_three_class_head():
    conf = {"not_out": 0.1, "full": 0.7, "partial": 0.2}
    assert (
        binary_confidence(conf, True) == 0.8999999999999999
        or round(binary_confidence(conf, True), 6) == 0.9
    )
    assert round(binary_confidence(conf, False), 6) == 0.1


def test_state_round_trips_through_json_shape():
    state = NotifyState(announced_is_out=True, pending_is_out=False, pending_since=T0)
    assert NotifyState.from_dict(state.to_dict()) == state
    assert NotifyState.from_dict(None) == NotifyState()

from collect.freshness import FeedState, update_feed_state

T0 = "2026-09-24T00:00:00Z"
T1 = "2026-09-24T00:15:00Z"
T2 = "2026-09-24T00:30:00Z"
T3 = "2026-09-24T00:45:00Z"

A = "a" * 64
B = "b" * 64


def test_first_frame_is_never_stale():
    state, stale = update_feed_state(FeedState(), A, T0)
    assert not stale
    assert state.repeat_count == 1
    assert state.first_seen == T0 and state.last_changed == T0


def test_three_identical_fetches_are_stale():
    state, stale = update_feed_state(FeedState(), A, T0)
    assert not stale
    state, stale = update_feed_state(state, A, T1)
    assert not stale and state.repeat_count == 2
    state, stale = update_feed_state(state, A, T2)
    assert stale and state.repeat_count == 3
    # first_seen still points at the frame that froze, for the site to show.
    assert state.first_seen == T0


def test_stale_stays_stale_until_the_bytes_change():
    state = FeedState(last_sha256=A, repeat_count=3, first_seen=T0, last_changed=T0)
    state, stale = update_feed_state(state, A, T1)
    assert stale and state.repeat_count == 4

    state, stale = update_feed_state(state, B, T2)
    assert not stale
    assert state.repeat_count == 1
    assert state.last_changed == T2


def test_threshold_is_configurable():
    state, stale = update_feed_state(FeedState(), A, T0, stale_after_repeats=2)
    assert not stale
    _, stale = update_feed_state(state, A, T1, stale_after_repeats=2)
    assert stale


def test_threshold_below_one_is_clamped_to_one():
    # A misconfigured 0 must not mean "every frame is stale forever" via a
    # count that can never reach it; it means "one repeat is enough".
    state, stale = update_feed_state(FeedState(), A, T0, stale_after_repeats=0)
    assert stale is True and state.repeat_count == 1


def test_state_round_trips_through_json_shape():
    state = FeedState(last_sha256=A, repeat_count=2, first_seen=T0, last_changed=T3)
    assert FeedState.from_dict(state.to_dict()) == state
    assert FeedState.from_dict(None) == FeedState()

// Debounced visibility-transition state machine — pure, so it can be tested
// without a Worker runtime (`npm test` in worker/).
//
// Why a state machine and not "compare this tick to the previous state.json":
// the raw predictions flap. Real history has out at 16:46, gone at 17:01, out
// again at 18:01 — three notifications for what a human would call one change.
// So a flip must survive TWO CONSECUTIVE ticks before it is announced: the
// first tick that disagrees with the announced state only arms `pending`, and
// the tick after it either confirms (announce) or clears it (flap, stay quiet).
//
// Cost of the debounce: one inference tick (~15 min) of latency on a genuine
// change. Benefit: the channel says "the mountain is out" only when it stays
// out, which is what makes the notification worth reacting to.

export interface NotifyState {
  /** Visibility the channel was last told about. null = never announced. */
  announced_is_out: boolean | null;
  /** A disagreeing observation awaiting confirmation on the next tick. */
  pending_is_out: boolean | null;
  /** When `pending_is_out` was armed (ISO 8601), for debugging. */
  pending_since: string | null;
}

export const INITIAL_NOTIFY_STATE: NotifyState = {
  announced_is_out: null,
  pending_is_out: null,
  pending_since: null,
};

export interface TransitionDecision {
  /** Announce this tick's visibility to the channel. */
  post: boolean;
  /** State to persist for the next tick. */
  next: NotifyState;
}

/**
 * Decide whether `isOut` is a confirmed change worth announcing.
 *
 * @param state  persisted state from the previous tick
 * @param isOut  this tick's prediction
 * @param now    ISO timestamp used to stamp a newly armed `pending`
 */
export function decideTransition(state: NotifyState, isOut: boolean, now: string): TransitionDecision {
  // First observation ever (or after the state object is lost): adopt the
  // current visibility silently. Announcing here would fire a notification on
  // every deploy that resets the object, which is noise, not news.
  if (state.announced_is_out === null) {
    return { post: false, next: { announced_is_out: isOut, pending_is_out: null, pending_since: null } };
  }

  // Agrees with what the channel already knows — nothing to say, and any
  // half-armed flip is now disproven.
  if (isOut === state.announced_is_out) {
    return { post: false, next: { ...state, pending_is_out: null, pending_since: null } };
  }

  // Disagrees, and the previous tick disagreed the same way: two in a row.
  if (state.pending_is_out === isOut) {
    return { post: true, next: { announced_is_out: isOut, pending_is_out: null, pending_since: null } };
  }

  // First tick of a disagreement — arm it and wait one more tick.
  return {
    post: false,
    next: { announced_is_out: state.announced_is_out, pending_is_out: isOut, pending_since: now },
  };
}

// What the channel says, and when — a pure state machine so it can be tested
// without a Worker runtime (`npm test` in worker/).
//
// The channel does two different jobs, and conflating them made it bad at both:
//
//   ALERT         "the mountain is out" — must be TRUSTWORTHY. You act on it.
//   LABEL REQUEST "I'm not sure — which is it?" — must be INFORMATIVE. You
//                 correct it, and the correction is worth a training row.
//
// They want opposite inputs. An alert is only worth posting when the model is
// confident; a label is only worth asking for when it ISN'T. Announcing every
// confirmed change sent the *least* reliable predictions out as alerts: over
// 2026-07-29/30, four of eight alerts fired below 0.71 binary confidence, one
// at 0.42, and the 18:15 "the mountain is out!" reversed 30 minutes later.
//
// So confidence routes the message. Two gates, in order:
//
//   1. DEBOUNCE (duration). A change must hold for two consecutive ticks.
//      Predictions flap — out 16:46, gone 17:01, out 18:01 is one change, not
//      three.
//   2. CONFIDENCE (certainty). A held change only ALERTS above
//      alertMinConfidence. Below it, `pending` stays armed: the alert is
//      delayed until the model is sure, never dropped.
//
// Anything the model is unsure about — transition or not — is a label-request
// candidate, rate-limited by a cooldown so a long ambiguous stretch asks once,
// not every 15 minutes. Confidence here is BINARY (p(out) = full + partial, vs
// p(not out)), because that is the question being asked.

export interface NotifyState {
  /** Visibility the channel was last told about. null = never announced. */
  announced_is_out: boolean | null;
  /** A disagreeing observation awaiting confirmation on the next tick. */
  pending_is_out: boolean | null;
  /** When `pending_is_out` was armed (ISO 8601), for debugging. */
  pending_since: string | null;
  /** When the last label request went out (ISO 8601) — drives the cooldown. */
  last_label_request: string | null;
}

export const INITIAL_NOTIFY_STATE: NotifyState = {
  announced_is_out: null,
  pending_is_out: null,
  pending_since: null,
  last_label_request: null,
};

export interface DecisionOptions {
  /** Binary confidence at or above which a held change is announced. */
  alertMinConfidence: number;
  /** Minimum gap between label requests, in milliseconds. */
  labelCooldownMs: number;
}

export const DEFAULT_OPTIONS: DecisionOptions = {
  alertMinConfidence: 0.85,
  labelCooldownMs: 4 * 60 * 60 * 1000,
};

/** `alert` announces a change; `label` asks which it is; `quiet` posts nothing. */
export type DecisionKind = "alert" | "label" | "quiet";

export interface TransitionDecision {
  kind: DecisionKind;
  /** State to persist for the next tick. */
  next: NotifyState;
}

/**
 * Route this tick to an alert, a label request, or silence.
 *
 * @param state       persisted state from the previous tick
 * @param isOut       this tick's prediction
 * @param confidence  BINARY confidence in `isOut` (p(out) or p(not out))
 * @param now         this tick's timestamp, ISO 8601
 */
export function decideTransition(
  state: NotifyState,
  isOut: boolean,
  confidence: number,
  now: string,
  options: DecisionOptions = DEFAULT_OPTIONS,
): TransitionDecision {
  const { alertMinConfidence, labelCooldownMs } = options;
  const confident = confidence >= alertMinConfidence;

  // A label request is due whenever the model is unsure and the cooldown has
  // elapsed. Evaluated last, so an alert always wins the tick.
  const labelDue = (): boolean => {
    if (confident) return false;
    if (!state.last_label_request) return true;
    const since = Date.parse(now) - Date.parse(state.last_label_request);
    return Number.isNaN(since) || since >= labelCooldownMs;
  };
  const asLabel = (next: NotifyState): TransitionDecision => ({
    kind: "label",
    next: { ...next, last_label_request: now },
  });

  // First observation ever (or after the state object is lost): adopt the
  // current visibility silently. Announcing here would fire on every deploy
  // that resets the object, which is noise, not news.
  if (state.announced_is_out === null) {
    const next = { ...state, announced_is_out: isOut, pending_is_out: null, pending_since: null };
    return labelDue() ? asLabel(next) : { kind: "quiet", next };
  }

  // Agrees with what the channel already knows — nothing to announce, and any
  // half-armed flip is now disproven.
  if (isOut === state.announced_is_out) {
    const next = { ...state, pending_is_out: null, pending_since: null };
    return labelDue() ? asLabel(next) : { kind: "quiet", next };
  }

  // Disagrees and held for a second consecutive tick. Announce it only if the
  // model is sure; otherwise keep pending armed so the alert is DELAYED to the
  // first confident tick rather than dropped — and meanwhile the frame is
  // exactly the kind worth asking about.
  if (state.pending_is_out === isOut && confident) {
    return {
      kind: "alert",
      next: { ...state, announced_is_out: isOut, pending_is_out: null, pending_since: null },
    };
  }

  // First tick of a disagreement, or a held-but-unconfident one: (re)arm.
  const next: NotifyState = {
    ...state,
    pending_is_out: isOut,
    pending_since: state.pending_is_out === isOut ? state.pending_since : now,
  };
  return labelDue() ? asLabel(next) : { kind: "quiet", next };
}

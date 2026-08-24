// Whether a broken pipeline says so out loud — a pure state machine, testable
// without a Worker runtime (`npm test` in worker/), like transition.ts.
//
// WHY THIS EXISTS
//
// On 2026-08-07 at 07:15Z the UW ATG webcam2 image (webcam2_latest.jpg) stopped
// existing upstream. Every */15 tick since has failed the same way:
//
//   container /predict returned 500: {"detail":"HTTPError: 404 Client Error:
//   Not Found for url: https://a.atmos.washington.edu/data/images/webcam2_latest.jpg"}
//
// 1,637 consecutive failures. Seventeen days. NOTHING WAS SAID. The tick's
// catch block appended the error to history.jsonl and called console.error,
// and that was the whole of it — a log nobody reads and an ndjson file nobody
// opens. The published state.json simply stopped moving, and the only symptom a
// human could see was a site quietly showing an hours-old reading.
//
// The Worker already had a Discord channel it posts to. It just never posted
// failures. That is the bug this module closes: an outage is now an event, not
// an absence of events.
//
// WHY NOT "ALERT ON EVERY FAILED TICK"
//
// A single failed tick is noise — the container cold-starts, METAR times out,
// a Durable Object gets reset mid-deploy. history.jsonl has 142 "container is
// not listening" and 37 NOAA read timeouts that all self-healed on the next
// tick. Paging on those trains you to ignore the channel, which is how you end
// up not noticing the one that lasts seventeen days.
//
// So two gates, mirroring transition.ts:
//
//   1. THRESHOLD. Say nothing until `alertAfterFailures` ticks have failed in a
//      row (default 4 = one hour of a dead pipeline). Transients never reach it.
//   2. COOLDOWN. Once alerting, re-alert at most every `realertCooldownMs`
//      (default 6h) so a long outage nags a few times a day rather than 96.
//
// And one thing that is NOT gated: RECOVERY. The first success after an alerted
// outage always posts, immediately. An all-clear is cheap, it is the message
// that closes the loop, and rate-limiting it would leave the channel showing a
// broken pipeline that has actually been fine for hours.

export interface HealthState {
  /** Consecutive failed ticks. Reset to 0 by any success. */
  consecutive_failures: number;
  /** When the current failure run began (ISO 8601). null = not failing. */
  failing_since: string | null;
  /** When the channel was last told about this run (ISO 8601). Drives cooldown. */
  last_alert: string | null;
  /** Whether the channel currently believes the pipeline is broken. */
  alerted: boolean;
}

export const INITIAL_HEALTH_STATE: HealthState = {
  consecutive_failures: 0,
  failing_since: null,
  last_alert: null,
  alerted: false,
};

export interface HealthOptions {
  /** Consecutive failures before the first alert. */
  alertAfterFailures: number;
  /** Minimum gap between repeat alerts within one outage, in milliseconds. */
  realertCooldownMs: number;
}

export const DEFAULT_HEALTH_OPTIONS: HealthOptions = {
  // Four ticks = one hour at the */15 cadence. Long enough that container cold
  // starts and upstream blips never reach it, short enough that a real outage
  // is in the channel within the hour instead of within a fortnight.
  alertAfterFailures: 4,
  realertCooldownMs: 6 * 60 * 60 * 1000,
};

/**
 * `down` announces (or re-announces) a sustained outage; `recovered` posts the
 * all-clear; `quiet` posts nothing.
 */
export type HealthKind = "down" | "recovered" | "quiet";

export interface HealthDecision {
  kind: HealthKind;
  next: HealthState;
  /**
   * Failure count carried on the message. On `recovered` this is the length of
   * the outage that just ended, not the (zero) current count — the all-clear
   * needs to say how bad it was.
   */
  failures: number;
  /** Start of the outage (ISO 8601), for "down since …" / "was down since …". */
  since: string | null;
}

/**
 * Fold one tick's outcome into the health state and decide what to say.
 *
 * @param state    Health bookkeeping as of the previous tick.
 * @param ok       Whether THIS tick succeeded.
 * @param now      Tick time, ISO 8601.
 * @param options  Threshold + cooldown.
 */
export function decideHealth(
  state: HealthState,
  ok: boolean,
  now: string,
  options: HealthOptions = DEFAULT_HEALTH_OPTIONS,
): HealthDecision {
  const { alertAfterFailures, realertCooldownMs } = options;

  if (ok) {
    // A success always clears the books. It only SPEAKS if the channel was
    // told the pipeline was broken — otherwise there is nothing to take back.
    const kind: HealthKind = state.alerted ? "recovered" : "quiet";
    return {
      kind,
      next: INITIAL_HEALTH_STATE,
      failures: state.consecutive_failures,
      since: state.failing_since,
    };
  }

  const consecutive = state.consecutive_failures + 1;
  // The run started at this tick only if it is the first failure of the run.
  const failingSince = state.failing_since ?? now;

  const belowThreshold = consecutive < alertAfterFailures;
  const cooling =
    state.alerted &&
    state.last_alert !== null &&
    Date.parse(now) - Date.parse(state.last_alert) < realertCooldownMs;

  if (belowThreshold || cooling) {
    return {
      kind: "quiet",
      next: {
        consecutive_failures: consecutive,
        failing_since: failingSince,
        last_alert: state.last_alert,
        alerted: state.alerted,
      },
      failures: consecutive,
      since: failingSince,
    };
  }

  return {
    kind: "down",
    next: {
      consecutive_failures: consecutive,
      failing_since: failingSince,
      last_alert: now,
      alerted: true,
    },
    failures: consecutive,
    since: failingSince,
  };
}

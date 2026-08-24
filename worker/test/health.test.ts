// Run with: npm test   (node:test + native TypeScript stripping, no deps)
//
// The regression these tests exist for: 1,637 consecutive failed ticks between
// 2026-08-07T07:15Z and 2026-08-24 produced zero Discord messages, because the
// error path only ever wrote to history.jsonl and console.error. The first test
// below is that outage, replayed.
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  DEFAULT_HEALTH_OPTIONS,
  INITIAL_HEALTH_STATE,
  decideHealth,
  type HealthKind,
  type HealthState,
} from "../src/health.ts";

const T0 = Date.parse("2026-08-07T07:15:00Z");
/** Tick n, 15 minutes apart — the real inference cadence. */
const at = (n: number) => new Date(T0 + n * 15 * 60_000).toISOString();

/** Feed a sequence of tick outcomes through the machine; one kind per tick. */
function run(outcomes: boolean[], from: HealthState = INITIAL_HEALTH_STATE): HealthKind[] {
  let state = from;
  return outcomes.map((ok, i) => {
    const { kind, next } = decideHealth(state, ok, at(i));
    state = next;
    return kind;
  });
}

const FAIL = false;
const OK = true;

test("the webcam outage: a sustained failure is announced, once, within the hour", () => {
  // Four ticks = the threshold. Nothing is said for the first three.
  const kinds = run(Array(4).fill(FAIL));
  assert.deepEqual(kinds, ["quiet", "quiet", "quiet", "down"]);
});

test("a day of the outage nags a few times, not 96", () => {
  // 96 ticks = 24 hours. With a 6h cooldown that is the first alert plus three
  // re-alerts. Before this module it was zero; the failure mode being guarded
  // against now is the opposite one.
  const kinds = run(Array(96).fill(FAIL));
  const downs = kinds.filter((k) => k === "down").length;
  assert.equal(downs, 4);
  assert.equal(kinds.filter((k) => k === "recovered").length, 0);
});

test("transients stay quiet — the container cold start that self-heals", () => {
  // history.jsonl has 142 "container is not listening" and 37 NOAA read
  // timeouts that recovered on the next tick. None of them should page.
  assert.deepEqual(run([FAIL, OK]), ["quiet", "quiet"]);
  assert.deepEqual(run([FAIL, FAIL, OK]), ["quiet", "quiet", "quiet"]);
  assert.deepEqual(run([FAIL, FAIL, FAIL, OK]), ["quiet", "quiet", "quiet", "quiet"]);
});

test("a success after an alerted outage always posts the all-clear, immediately", () => {
  const kinds = run([...Array(4).fill(FAIL), OK]);
  assert.deepEqual(kinds, ["quiet", "quiet", "quiet", "down", "recovered"]);
});

test("recovery is not rate-limited — it fires inside the re-alert cooldown", () => {
  // Alert at tick 3, recover at tick 4: fifteen minutes, far inside the 6h
  // cooldown. An all-clear the channel never hears is worse than a chatty one.
  const kinds = run([...Array(4).fill(FAIL), OK]);
  assert.equal(kinds.at(-1), "recovered");
});

test("recovery reports the outage it ended, not the current (zero) count", () => {
  let state: HealthState = INITIAL_HEALTH_STATE;
  for (let i = 0; i < 10; i++) state = decideHealth(state, FAIL, at(i)).next;

  const decision = decideHealth(state, OK, at(10));
  assert.equal(decision.kind, "recovered");
  assert.equal(decision.failures, 10);
  assert.equal(decision.since, at(0));
  assert.deepEqual(decision.next, INITIAL_HEALTH_STATE);
});

test("the failure run is stamped at its first tick, not its alerting tick", () => {
  let state: HealthState = INITIAL_HEALTH_STATE;
  for (let i = 0; i < 4; i++) state = decideHealth(state, FAIL, at(i)).next;
  assert.equal(state.failing_since, at(0));
  assert.equal(state.last_alert, at(3));
});

test("a flapping pipeline re-arms — recovery resets the threshold", () => {
  // Fail to the alert, recover, then fail again: the second outage must earn
  // its own four ticks rather than inheriting the first run's count.
  const kinds = run([...Array(4).fill(FAIL), OK, FAIL, FAIL, FAIL, FAIL]);
  assert.deepEqual(kinds, [
    "quiet",
    "quiet",
    "quiet",
    "down",
    "recovered",
    "quiet",
    "quiet",
    "quiet",
    "down",
  ]);
});

test("a success with nothing to take back stays quiet", () => {
  assert.deepEqual(run([OK, OK, OK]), ["quiet", "quiet", "quiet"]);
});

test("thresholds are options, not constants", () => {
  const eager = { alertAfterFailures: 1, realertCooldownMs: 0 };
  let state: HealthState = INITIAL_HEALTH_STATE;
  const kinds = [FAIL, FAIL].map((ok) => {
    const d = decideHealth(state, ok, at(0), eager);
    state = d.next;
    return d.kind;
  });
  assert.deepEqual(kinds, ["down", "down"]);
});

test("defaults are the documented one-hour / six-hour policy", () => {
  assert.equal(DEFAULT_HEALTH_OPTIONS.alertAfterFailures, 4);
  assert.equal(DEFAULT_HEALTH_OPTIONS.realertCooldownMs, 6 * 60 * 60 * 1000);
});

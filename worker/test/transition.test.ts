// Run with: npm test   (node:test + native TypeScript stripping, no deps)
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  DEFAULT_OPTIONS,
  INITIAL_NOTIFY_STATE,
  decideTransition,
  type DecisionKind,
  type NotifyState,
} from "../src/transition.ts";

const SURE = 0.95;
const UNSURE = 0.6;
const T0 = Date.parse("2026-07-29T00:00:00Z");
/** Tick n, 15 minutes apart — the real inference cadence. */
const at = (n: number) => new Date(T0 + n * 15 * 60_000).toISOString();

interface Tick {
  isOut: boolean;
  confidence: number;
}

/** Feed a tick sequence through the machine; return one kind per tick. */
function run(ticks: Tick[], from: NotifyState = INITIAL_NOTIFY_STATE): DecisionKind[] {
  let state = from;
  const kinds: DecisionKind[] = [];
  ticks.forEach((tick, i) => {
    const { kind, next } = decideTransition(state, tick.isOut, tick.confidence, at(i));
    kinds.push(kind);
    state = next;
  });
  return kinds;
}

const out = (confidence = SURE): Tick => ({ isOut: true, confidence });
const gone = (confidence = SURE): Tick => ({ isOut: false, confidence });
const announced = (isOut: boolean, last_label_request: string | null = at(0)): NotifyState => ({
  announced_is_out: isOut,
  pending_is_out: null,
  pending_since: null,
  last_label_request,
});

// ---------- Debounce (duration gate) ----------

test("first observation is adopted silently, not announced", () => {
  const { kind, next } = decideTransition(INITIAL_NOTIFY_STATE, true, SURE, at(0));
  assert.equal(kind, "quiet");
  assert.equal(next.announced_is_out, true);
});

test("a confident change announces only after it survives a second tick", () => {
  assert.deepEqual(run([out(), out()], announced(false)), ["quiet", "alert"]);
});

test("a single-tick flap is swallowed", () => {
  // The real 2026-07-24 sequence: out, gone for one tick, out again.
  assert.deepEqual(run([out(), out(), gone(), out(), out()], announced(false)), [
    "quiet",
    "alert",
    "quiet",
    "quiet",
    "quiet",
  ]);
});

test("both directions announce, and a held state is not re-announced", () => {
  const kinds = run([out(), out(), out(), gone(), gone(), gone()], announced(false));
  assert.deepEqual(kinds, ["quiet", "alert", "quiet", "quiet", "alert", "quiet"]);
});

// ---------- Confidence gate ----------

test("an unconfident change is NOT announced", () => {
  const kinds = run([out(UNSURE), out(UNSURE), out(UNSURE)], announced(false));
  assert.equal(kinds.filter((k) => k === "alert").length, 0);
});

test("an unconfident change is delayed, not dropped — it fires when the model is sure", () => {
  // Held unconfidently for three ticks, then one confident tick.
  const kinds = run([out(UNSURE), out(UNSURE), out(UNSURE), out(SURE)], announced(false));
  assert.deepEqual(kinds.slice(-1), ["alert"], "the confident tick announces the pending change");
});

test("pending_since marks the first disagreement, not the confident one", () => {
  let state = announced(false);
  const first = decideTransition(state, true, UNSURE, at(1));
  state = first.next;
  const second = decideTransition(state, true, UNSURE, at(2));
  assert.equal(second.next.pending_since, at(1), "re-arming must not reset the clock");
});

// ---------- Label requests ----------

test("an unsure tick asks for a label", () => {
  const { kind, next } = decideTransition(announced(true, null), true, UNSURE, at(1));
  assert.equal(kind, "label");
  assert.equal(next.last_label_request, at(1));
});

test("a confident tick never asks for a label", () => {
  assert.equal(decideTransition(announced(true, null), true, SURE, at(1)).kind, "quiet");
});

test("the cooldown suppresses repeat asks during a long ambiguous stretch", () => {
  // 17 unsure ticks = 4h. With a 4h cooldown: ask on the first, then once more
  // when 4h has elapsed — not 17 times.
  const kinds = run(
    Array.from({ length: 17 }, () => out(UNSURE)),
    announced(true, null),
  );
  assert.equal(
    kinds.filter((k) => k === "label").length,
    2,
    "one ask up front, one after the cooldown elapses",
  );
});

test("an alert always wins the tick over a label request", () => {
  // Confident enough to alert ⇒ by definition not unsure ⇒ never both.
  const state: NotifyState = { ...announced(false, null), pending_is_out: true };
  assert.equal(decideTransition(state, true, SURE, at(1)).kind, "alert");
});

// ---------- Regression: the real 2026-07-29/30 tick sequence ----------

test("replays yesterday: noisy alerts become a few trustworthy ones", () => {
  // (isOut, binary confidence) for ok ticks from the deploy onward, taken from
  // history.jsonl. Under the old duration-only rule this stretch alerted seven
  // times, four of them below 0.71 — including an "out" at 18:15 that reversed
  // 30 minutes later.
  const real: Tick[] = [
    gone(0.5), out(0.69), out(0.84), gone(0.69), gone(0.88), gone(0.55), out(0.65),
    out(0.67), gone(0.85), gone(1.0), gone(0.99), gone(0.99), gone(0.84), gone(0.96),
    gone(0.91), gone(0.77), gone(0.46), out(0.71), out(0.66), out(0.66), out(0.92),
    out(0.76), out(0.58), out(0.98), gone(0.52), out(0.64), out(0.78), out(0.82),
    out(0.67), out(0.95), gone(0.42), gone(0.42), gone(0.64), out(0.91), out(0.66),
    out(0.89), out(1.0), out(0.95), out(0.65), out(0.98), out(0.98), gone(0.5),
    out(0.99), out(0.99), out(1.0), out(1.0), out(0.84), gone(1.0), gone(1.0),
  ];
  const kinds = run(real, announced(true, null));
  const alerts = kinds.filter((k) => k === "alert").length;
  const labels = kinds.filter((k) => k === "label").length;

  assert.ok(alerts <= 4, `only confident changes announce, got ${alerts}`);
  assert.ok(labels >= 2 && labels <= 6, `label asks stay low-noise, got ${labels}`);
  assert.ok(alerts + labels <= 10, `total volume stays low, got ${alerts + labels}`);
});

test("options are honoured — a stricter gate announces less", () => {
  const strict = { ...DEFAULT_OPTIONS, alertMinConfidence: 0.99 };
  let state = announced(false, at(0));
  const kinds: DecisionKind[] = [];
  [out(0.95), out(0.95)].forEach((t, i) => {
    const d = decideTransition(state, t.isOut, t.confidence, at(i + 1), strict);
    kinds.push(d.kind);
    state = d.next;
  });
  assert.equal(kinds.filter((k) => k === "alert").length, 0);
});

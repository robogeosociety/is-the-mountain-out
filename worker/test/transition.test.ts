// Run with: npm test   (node:test + native TypeScript stripping, no deps)
import assert from "node:assert/strict";
import { test } from "node:test";

import { INITIAL_NOTIFY_STATE, decideTransition, type NotifyState } from "../src/transition.ts";

const NOW = "2026-07-28T00:00:00Z";

/** Feed a tick sequence through the machine; return the ticks that announced. */
function run(ticks: boolean[], from: NotifyState = INITIAL_NOTIFY_STATE): boolean[] {
  let state = from;
  const announced: boolean[] = [];
  for (const isOut of ticks) {
    const { post, next } = decideTransition(state, isOut, NOW);
    if (post) announced.push(isOut);
    state = next;
  }
  return announced;
}

test("first observation is adopted silently, not announced", () => {
  const { post, next } = decideTransition(INITIAL_NOTIFY_STATE, true, NOW);
  assert.equal(post, false);
  assert.equal(next.announced_is_out, true);
});

test("a change announces only after it survives a second tick", () => {
  assert.deepEqual(run([false, true]), [], "one disagreeing tick only arms pending");
  assert.deepEqual(run([false, true, true]), [true]);
});

test("a single-tick flap is swallowed", () => {
  const notOut: NotifyState = { announced_is_out: false, pending_is_out: null, pending_since: null };
  // The real 2026-07-24 sequence: out, gone for one tick, out again. The old
  // compare-to-previous-tick logic posted three times; this posts once.
  assert.deepEqual(run([true, true, false, true, true], notOut), [true]);
});

test("both directions announce", () => {
  assert.deepEqual(run([false, true, true, false, false]), [true, false]);
});

test("a confirmed state is not re-announced while it holds", () => {
  assert.deepEqual(run([false, true, true, true, true, true]), [true]);
});

test("pending is cleared once the flip is disproven", () => {
  const armed = decideTransition({ announced_is_out: false, pending_is_out: null, pending_since: null }, true, NOW);
  assert.equal(armed.next.pending_is_out, true);
  assert.equal(armed.next.pending_since, NOW);

  const disproven = decideTransition(armed.next, false, NOW);
  assert.equal(disproven.post, false);
  assert.equal(disproven.next.pending_is_out, null);
  assert.equal(disproven.next.pending_since, null);
  assert.equal(disproven.next.announced_is_out, false);
});

test("a lost state object re-adopts silently instead of firing", () => {
  assert.deepEqual(run([true, true], INITIAL_NOTIFY_STATE), []);
});

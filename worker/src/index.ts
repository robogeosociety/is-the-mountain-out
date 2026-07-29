// Cloudflare Worker that owns the */15 inference cadence.
//
// Cron handler:
//   1. Calls the bound container's /predict endpoint.
//   2. Wraps the call in a try/finally that produces one structured record per
//      tick (status: "ok" with state, or status: "error" with message).
//   3. On success: overwrites state.json in the public R2 bucket.
//   4. Always: appends a record to history.jsonl in R2 (GET, append, PUT —
//      R2 has no append op, but at one record per 15min the file stays small).
//
// The container is passed R2 credentials as env vars so it can pull the
// latest checkpoint from R2 on first /predict call. Worker writes the
// inference output to a separate public R2 bucket the SPA reads from
// directly — no GitHub Contents API in the inference path.

import { Container, getContainer } from "@cloudflare/containers";
import { notifyDiscordTest, notifyMountainVisibility } from "./discord-mountain-notify";
import { INITIAL_NOTIFY_STATE, decideTransition, type NotifyState } from "./transition";

interface Env {
  INFERENCE_CONTAINER: DurableObjectNamespace<InferenceContainer>;
  STATE_BUCKET: R2Bucket;
  CAPTURE_BUCKET: R2Bucket;
  R2_ACCESS_KEY_ID: string;
  R2_SECRET_ACCESS_KEY: string;
  // Credentials the Worker posts visibility changes with — as the labeler bot,
  // so the posts are reaction-labelable (see discord-mountain-notify.ts).
  // Optional so a missing secret degrades to a logged skip rather than a thrown
  // error. Set out-of-band with `wrangler secret put`.
  DISCORD_BOT_TOKEN?: string;
  DISCORD_CHANNEL_ID?: string;
}

export class InferenceContainer extends Container<Env> {
  defaultPort = 8080;
  sleepAfter = "5m";

  override envVars = {
    R2_ACCESS_KEY_ID: this.env.R2_ACCESS_KEY_ID,
    R2_SECRET_ACCESS_KEY: this.env.R2_SECRET_ACCESS_KEY,
  };
}

interface PredictionState {
  timestamp_utc: string;
  class_index: number;
  class_name: string;
  is_out: boolean;
  confidence: Record<string, number>;
  weather: Record<string, unknown>;
  webcam_url: string;
  model_version: string | null;
}

type HistoryRecord =
  | {
      started_at: string;
      finished_at: string;
      duration_seconds: number;
      status: "ok";
      state: PredictionState;
    }
  | {
      started_at: string;
      finished_at: string;
      duration_seconds: number;
      status: "error";
      error: { type: string; message: string };
    };

const STATE_KEY = "state.json";
const HISTORY_KEY = "history.jsonl";
// Notification bookkeeping (what the channel was last told, plus any flip
// awaiting confirmation). Separate from state.json so the SPA's contract stays
// exactly the published prediction — see transition.ts for the state machine.
const NOTIFY_STATE_KEY = "notify-state.json";

const isoUtc = (d: Date) => d.toISOString().replace(/\.\d{3}Z$/, "Z");

async function callContainer(env: Env): Promise<PredictionState> {
  const stub = getContainer(env.INFERENCE_CONTAINER);
  const resp = await stub.fetch("http://container/predict", { method: "POST" });
  if (!resp.ok) {
    const body = await resp.text().catch(() => "<unreadable>");
    throw new Error(`container /predict returned ${resp.status}: ${body.slice(0, 500)}`);
  }
  return (await resp.json()) as PredictionState;
}

async function publishState(env: Env, state: PredictionState): Promise<void> {
  await env.STATE_BUCKET.put(STATE_KEY, JSON.stringify(state, null, 2) + "\n", {
    httpMetadata: { contentType: "application/json" },
  });
}

async function getNotifyState(env: Env): Promise<NotifyState> {
  try {
    const obj = await env.STATE_BUCKET.get(NOTIFY_STATE_KEY);
    if (!obj) return INITIAL_NOTIFY_STATE;
    return { ...INITIAL_NOTIFY_STATE, ...(JSON.parse(await obj.text()) as Partial<NotifyState>) };
  } catch (e) {
    // Unreadable bookkeeping re-adopts the current visibility silently rather
    // than announcing it — a corrupt object must not become a false alert.
    console.warn("failed to read notify state; re-adopting current visibility:", e);
    return INITIAL_NOTIFY_STATE;
  }
}

async function putNotifyState(env: Env, state: NotifyState): Promise<void> {
  await env.STATE_BUCKET.put(NOTIFY_STATE_KEY, JSON.stringify(state, null, 2) + "\n", {
    httpMetadata: { contentType: "application/json" },
  });
}

// METAR source for persisted transition captures — same NOAA endpoint and
// station the collector's WeatherFetcher uses (mountain.toml [weather]).
const METAR_URL = "https://tgftp.nws.noaa.gov/data/observations/metar/stations/KSEA.TXT";

// Persist the frame (and METAR) a transition notification announces, under the
// collector's standard capture-key layout, so a reaction on the notification
// can label a real stored capture (the labeler bot gates on the key existing).
// Returns the key AND the bytes — the notification embeds these exact bytes as
// an attachment, so what you react to is what gets labeled. Null on failure:
// the notification then falls back to a prose footer and isn't labelable.
async function persistTransitionCapture(
  env: Env,
  state: PredictionState,
): Promise<{ key: string; frame: ArrayBuffer } | null> {
  try {
    const frameResp = await fetch(state.webcam_url);
    if (!frameResp.ok) throw new Error(`webcam fetch returned ${frameResp.status}`);
    const frame = await frameResp.arrayBuffer();

    const now = new Date();
    const pad = (n: number, width = 2) => String(n).padStart(width, "0");
    const dateStr = `${now.getUTCFullYear()}${pad(now.getUTCMonth() + 1)}${pad(now.getUTCDate())}`;
    const timeStr =
      `${pad(now.getUTCHours())}${pad(now.getUTCMinutes())}${pad(now.getUTCSeconds())}` +
      `_${pad(now.getUTCMilliseconds() * 1000, 6)}_UTC`;
    const captureDir = `${dateStr}/${timeStr}`;
    const imageKey = `${captureDir}/images/${timeStr}_worker_transition.jpg`;

    await env.CAPTURE_BUCKET.put(imageKey, frame, {
      httpMetadata: { contentType: "image/jpeg" },
    });

    // METAR is best-effort: labeled samples want the paired weather vector,
    // but a missing metar.txt must not cost us the labelable image.
    try {
      const metarResp = await fetch(METAR_URL);
      if (metarResp.ok) {
        const lines = (await metarResp.text()).trim().split("\n");
        const metar = lines.length >= 2 ? lines[1] : lines[0];
        await env.CAPTURE_BUCKET.put(`${captureDir}/metar/metar.txt`, metar, {
          httpMetadata: { contentType: "text/plain" },
        });
      }
    } catch (e) {
      console.warn("transition METAR persist failed (image kept):", e);
    }
    return { key: imageKey, frame };
  } catch (e) {
    console.error("transition capture persist failed; notification will not be labelable:", e);
    return null;
  }
}

// Announce a confirmed change in BOTH directions — the channel is the live
// answer to "is the mountain out?", so it has to say when the answer stops
// being yes as well as when it starts. `decideTransition` supplies the
// two-consecutive-ticks debounce that keeps flapping predictions off the wire.
async function notifyTransition(env: Env, next: PredictionState): Promise<void> {
  const { post, next: notifyState } = decideTransition(
    await getNotifyState(env),
    next.is_out,
    isoUtc(new Date()),
  );
  // Persist BEFORE posting: a duplicate alert is worse than a missed one, and
  // a crash between the two would otherwise re-announce on the next tick.
  await putNotifyState(env, notifyState);
  if (!post) return;

  const visible = next.is_out;
  const confidence = visible
    ? (next.confidence?.full ?? 0) + (next.confidence?.partial ?? 0)
    : (next.confidence?.not_out ?? 0);
  const label = visible ? (next.class_name === "full" ? "Full" : "Partial") : "Not Out";
  const capture = await persistTransitionCapture(env, next);
  await notifyMountainVisibility(env, {
    visible,
    confidence,
    label,
    timestamp: next.timestamp_utc,
    frame: capture?.frame,
    captureKey: capture?.key,
    imageUrl: next.webcam_url,
  });
}

async function sendNotificationTest(env: Env): Promise<void> {
  await notifyDiscordTest(env);
}

async function appendHistory(env: Env, record: HistoryRecord): Promise<void> {
  const existing = await env.STATE_BUCKET.get(HISTORY_KEY);
  const prev = existing ? await existing.text() : "";
  const next = prev + JSON.stringify(record) + "\n";
  await env.STATE_BUCKET.put(HISTORY_KEY, next, {
    httpMetadata: { contentType: "application/x-ndjson" },
  });
}

async function tick(env: Env): Promise<void> {
  const startedAt = new Date();
  let record: HistoryRecord;
  try {
    const state = await callContainer(env);
    await publishState(env, state);
    await notifyTransition(env, state);
    const finishedAt = new Date();
    record = {
      started_at: isoUtc(startedAt),
      finished_at: isoUtc(finishedAt),
      duration_seconds: Number(((finishedAt.getTime() - startedAt.getTime()) / 1000).toFixed(3)),
      status: "ok",
      state,
    };
  } catch (err) {
    const finishedAt = new Date();
    const e = err as Error;
    record = {
      started_at: isoUtc(startedAt),
      finished_at: isoUtc(finishedAt),
      duration_seconds: Number(((finishedAt.getTime() - startedAt.getTime()) / 1000).toFixed(3)),
      status: "error",
      error: { type: e.name || "Error", message: e.message ?? String(err) },
    };
    console.error("inference tick failed:", e);
  }
  await appendHistory(env, record);
}

export default {
  async scheduled(_event: ScheduledController, env: Env, ctx: ExecutionContext) {
    ctx.waitUntil(tick(env));
  },

  async fetch(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(req.url);
    if (url.pathname === "/run" && req.method === "POST") {
      ctx.waitUntil(tick(env));
      return new Response("queued\n", { status: 202 });
    }
    if (url.pathname === "/notify-test" && req.method === "POST") {
      ctx.waitUntil(sendNotificationTest(env));
      return new Response("test notification queued\n", { status: 202 });
    }
    return new Response("mountain-inference worker\n", { status: 200 });
  },
} satisfies ExportedHandler<Env>;

// Discord webhook notification module for is-the-mountain-out.
//
// Replaces the legacy ntfy.sh push path. The Worker calls:
//   - notifyMountainVisibility(env, result) on the Not Out -> visible transition
//   - notifyDiscordTest(env)               from the /notify-test endpoint
//
// Integration (worker/src/index.ts):
//   import { notifyMountainVisibility } from "./discord-mountain-notify";
//   ctx.waitUntil(notifyMountainVisibility(env, result));
//
// Delivery never throws — webhook failures are logged and swallowed so they
// cannot crash the scheduled handler or block the history append.

export interface PredictionResult {
  visible: boolean;
  confidence: number; // 0..1, combined visible-class confidence
  label: string; // "Full" | "Partial" | raw class name
  imageUrl?: string; // webcam snapshot to embed
  timestamp?: string; // ISO 8601; defaults to now
  // R2 key of the persisted frame this notification announces. When set it
  // becomes the embed footer verbatim — the contract the labeler bot parses so
  // a 👍/⛅/👎 reaction on the notification records a training label.
  captureKey?: string;
}

export interface Env {
  // Discord webhook URL. Optional so a missing secret degrades to a logged
  // skip rather than a thrown error (mirrors the old NTFY_TOPIC guard).
  DISCORD_WEBHOOK_URL?: string;
}

const COLOR_VISIBLE = 0x2ecc71; // green
const COLOR_NOT_VISIBLE = 0x95a5a6; // gray
const COLOR_TEST = 0x3498db; // blue

interface DiscordEmbed {
  title: string;
  description?: string;
  color: number;
  fields?: { name: string; value: string; inline?: boolean }[];
  image?: { url: string };
  footer?: { text: string };
  timestamp?: string;
}

// Single delivery path for every embed. Guards on a missing webhook, logs
// non-2xx responses with a truncated body, and swallows network errors.
async function postWebhook(env: Env, embed: DiscordEmbed, context: string): Promise<boolean> {
  if (!env.DISCORD_WEBHOOK_URL) {
    console.warn(`DISCORD_WEBHOOK_URL not set; skipping Discord ${context}`);
    return false;
  }
  try {
    const response = await fetch(env.DISCORD_WEBHOOK_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ embeds: [embed] }),
    });
    if (!response.ok) {
      const body = await response.text().catch(() => "(unreadable body)");
      console.error(
        `Discord ${context} failed: ${response.status} ${response.statusText} — ${body.slice(0, 200)}`,
      );
      return false;
    }
    return true;
  } catch (err) {
    console.error(
      `Discord ${context} request error:`,
      err instanceof Error ? err.message : String(err),
    );
    return false;
  }
}

/**
 * Post a Discord embed announcing the mountain visibility prediction.
 * Returns true if the webhook was delivered successfully, false otherwise.
 */
export async function notifyMountainVisibility(env: Env, result: PredictionResult): Promise<boolean> {
  const { visible, confidence, label, imageUrl, timestamp, captureKey } = result;
  const confidencePct = (confidence * 100).toFixed(1);
  const ts = timestamp ?? new Date().toISOString();

  const embed: DiscordEmbed = {
    title: visible ? "🏔️ The mountain is out!" : "☁️ The mountain is not visible",
    color: visible ? COLOR_VISIBLE : COLOR_NOT_VISIBLE,
    fields: [
      { name: "Confidence", value: `${confidencePct}%`, inline: true },
      { name: "Classification", value: label, inline: true },
    ],
    // The capture key makes the notification labelable (react 👍/⛅/👎 to
    // correct or confirm the model); prose footer = not labelable.
    footer: { text: captureKey ?? "is-the-mountain-out • Automated prediction" },
    timestamp: ts,
  };
  if (imageUrl) {
    embed.image = { url: imageUrl };
  }

  return postWebhook(env, embed, "visibility notification");
}

/**
 * Post a one-shot Discord test embed (no inference involved) so the
 * Worker -> Discord webhook path can be verified end to end.
 */
export async function notifyDiscordTest(env: Env): Promise<boolean> {
  const embed: DiscordEmbed = {
    title: "🏔️ Worker notification test",
    description:
      "Test from the mountain-inference Worker. If you see this in Discord, the Worker → Discord webhook path is healthy.",
    color: COLOR_TEST,
    footer: { text: "is-the-mountain-out • Notification test" },
    timestamp: new Date().toISOString(),
  };
  return postWebhook(env, embed, "test notification");
}

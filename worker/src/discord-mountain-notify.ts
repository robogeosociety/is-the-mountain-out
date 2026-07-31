// Discord notification module for is-the-mountain-out.
//
// The Worker calls:
//   - notifyMountainVisibility(env, result) on a confirmed visibility change
//   - notifyDiscordTest(env)                from the /notify-test endpoint
//
// WHY A BOT TOKEN AND NOT A WEBHOOK
// ---------------------------------
// These notifications are the project's feedback surface: you react 👍/⛅/👎 to
// confirm or correct the call, and the labeler bot (BOT.md) turns that into a
// training label keyed on the announced frame. That only works if the bot can
// READ the message's embed footer, which carries the capture key.
//
// A bot without the privileged Message Content intent gets `embeds: []` for
// messages authored by anyone else — including a webhook. So while the Worker
// posted via `DISCORD_WEBHOOK_URL`, every reaction on a notification was
// silently dropped. Posting with the labeler's own bot token makes the
// notification a message the bot authored, which Discord always delivers in
// full. No privileged intent, and the bot can also edit its 🏷️ Label
// acknowledgment onto the post.
//
// The announced frame is uploaded as an ATTACHMENT rather than linked as
// `webcam_url`: that URL is `…/webcam2_latest.jpg`, so a linked image silently
// becomes a different picture later. You must be labeling the frame you can
// see.
//
// Delivery never throws — failures are logged and swallowed so they cannot
// crash the scheduled handler or block the history append.

const DISCORD_API = "https://discord.com/api/v10";

export interface PredictionResult {
  visible: boolean;
  confidence: number; // 0..1, confidence in the announced state
  label: string; // "Full" | "Partial" | "Not Out"
  timestamp?: string; // ISO 8601; defaults to now
  // The announced frame, persisted to R2 by the caller. Both are required for
  // the post to be labelable: the bytes so the embed shows the frame that was
  // classified, the key so a reaction can be recorded against it.
  frame?: ArrayBuffer;
  captureKey?: string;
  // Fallback image when no frame was persisted (post is then not labelable).
  imageUrl?: string;
}

export interface Env {
  // Both optional so a missing secret degrades to a logged skip rather than a
  // thrown error. Set with `wrangler secret put` — see NOTIFICATIONS.md.
  DISCORD_BOT_TOKEN?: string;
  DISCORD_CHANNEL_ID?: string;
}

const COLOR_VISIBLE = 0x2ecc71; // green
const COLOR_NOT_VISIBLE = 0x95a5a6; // gray
const COLOR_TEST = 0x3498db; // blue

// The label vocabulary, mirroring bot/labeler.py EMOJI_TO_CLASS. Seeded onto
// every labelable post in this order, and rendered as the legend below — which
// matches bot/labeler.py reaction_legend() (ordered by class index, descending)
// so both surfaces read identically.
const LABEL_REACTIONS = [
  { emoji: "⛅", label: "Partial" },
  { emoji: "👍", label: "Full" },
  { emoji: "👎", label: "Not Out" },
];
const REACTION_LEGEND = LABEL_REACTIONS.map((r) => `${r.emoji} ${r.label}`).join(" · ");

interface DiscordEmbed {
  title: string;
  description?: string;
  color: number;
  fields?: { name: string; value: string; inline?: boolean }[];
  image?: { url: string };
  footer?: { text: string };
  timestamp?: string;
}

function botHeaders(env: Env): Record<string, string> {
  return { Authorization: `Bot ${env.DISCORD_BOT_TOKEN}` };
}

/**
 * Post one embed to the configured channel as the bot, optionally attaching a
 * JPEG frame. Returns the new message id, or null if the post did not land.
 */
async function postAsBot(
  env: Env,
  embed: DiscordEmbed,
  context: string,
  frame?: ArrayBuffer,
): Promise<string | null> {
  if (!env.DISCORD_BOT_TOKEN || !env.DISCORD_CHANNEL_ID) {
    console.warn(`DISCORD_BOT_TOKEN/DISCORD_CHANNEL_ID not set; skipping Discord ${context}`);
    return null;
  }
  try {
    let body: BodyInit;
    const headers = botHeaders(env);
    if (frame) {
      // Multipart: the embed references the upload by name. Content-Type is
      // left unset so fetch writes the multipart boundary itself.
      const form = new FormData();
      form.append("payload_json", JSON.stringify({ embeds: [embed] }));
      form.append("files[0]", new Blob([frame], { type: "image/jpeg" }), "capture.jpg");
      body = form;
    } else {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify({ embeds: [embed] });
    }

    const response = await fetch(`${DISCORD_API}/channels/${env.DISCORD_CHANNEL_ID}/messages`, {
      method: "POST",
      headers,
      body,
    });
    if (!response.ok) {
      const text = await response.text().catch(() => "(unreadable body)");
      console.error(
        `Discord ${context} failed: ${response.status} ${response.statusText} — ${text.slice(0, 200)}`,
      );
      return null;
    }
    const message = (await response.json()) as { id?: string };
    return message.id ?? null;
  } catch (err) {
    console.error(
      `Discord ${context} request error:`,
      err instanceof Error ? err.message : String(err),
    );
    return null;
  }
}

/**
 * Seed the three label reactions so the post is one tap away from a label.
 *
 * Sequential by necessity — Discord rate-limits reaction adds per channel, and
 * a 429 is honored once before giving up on that emoji. Best-effort: a missing
 * seed reaction costs a tap, not the label (the bot records any human reaction
 * on the message).
 */
async function seedLabelReactions(env: Env, messageId: string): Promise<void> {
  const base = `${DISCORD_API}/channels/${env.DISCORD_CHANNEL_ID}/messages/${messageId}/reactions`;
  for (const { emoji } of LABEL_REACTIONS) {
    const url = `${base}/${encodeURIComponent(emoji)}/@me`;
    try {
      let response = await fetch(url, { method: "PUT", headers: botHeaders(env) });
      if (response.status === 429) {
        const retry = (await response.json().catch(() => ({}))) as { retry_after?: number };
        await new Promise((resolve) => setTimeout(resolve, (retry.retry_after ?? 1) * 1000));
        response = await fetch(url, { method: "PUT", headers: botHeaders(env) });
      }
      if (!response.ok) {
        console.warn(`seeding reaction ${emoji} failed: ${response.status} ${response.statusText}`);
      }
    } catch (err) {
      console.warn(`seeding reaction ${emoji} errored:`, err instanceof Error ? err.message : String(err));
    }
  }
}

/**
 * Announce a confirmed visibility change and make it labelable.
 * Returns true if the message was posted.
 */
export async function notifyMountainVisibility(env: Env, result: PredictionResult): Promise<boolean> {
  const { visible, confidence, label, timestamp, frame, captureKey, imageUrl } = result;
  const labelable = Boolean(frame && captureKey);

  const fields = [
    { name: "Confidence", value: `${(confidence * 100).toFixed(1)}%`, inline: true },
    { name: "Classification", value: label, inline: true },
  ];
  if (labelable) {
    fields.push({ name: "Right or wrong?", value: REACTION_LEGEND, inline: false });
  }

  const embed: DiscordEmbed = {
    title: visible ? "🏔️ The mountain is out!" : "☁️ The mountain is gone",
    color: visible ? COLOR_VISIBLE : COLOR_NOT_VISIBLE,
    fields,
    // The capture key IS the footer — the contract bot/labeler.py parses. A
    // prose footer means the frame never persisted, so nothing is labelable.
    footer: { text: captureKey ?? "is-the-mountain-out • Automated prediction" },
    timestamp: timestamp ?? new Date().toISOString(),
  };
  if (labelable) {
    embed.image = { url: "attachment://capture.jpg" };
  } else if (imageUrl) {
    embed.image = { url: imageUrl };
  }

  const messageId = await postAsBot(env, embed, "visibility notification", labelable ? frame : undefined);
  if (messageId === null) return false;
  if (labelable) await seedLabelReactions(env, messageId);
  return true;
}

const COLOR_UNSURE = 0xf1c40f; // amber — a question, not an announcement

/**
 * Ask which it is, when the model can't tell.
 *
 * Deliberately NOT styled as an alert: no "the mountain is out!" claim, amber
 * rather than green/gray, and the confidence is shown so a low number reads as
 * the reason you're being asked. These frames are the ones a label teaches the
 * model most — the confident ones it already gets right.
 *
 * Unlike an alert, this is pointless without a labelable frame: the whole post
 * IS the question. No frame ⇒ no post.
 */
export async function notifyLabelRequest(env: Env, result: PredictionResult): Promise<boolean> {
  const { visible, confidence, label, timestamp, frame, captureKey } = result;
  if (!frame || !captureKey) {
    console.warn("label request skipped: no persisted frame to ask about");
    return false;
  }

  const embed: DiscordEmbed = {
    title: "🤔 Is the mountain out?",
    description: "The model isn't sure about this one. Your answer becomes a training label.",
    color: COLOR_UNSURE,
    fields: [
      { name: "Best guess", value: label, inline: true },
      { name: "Confidence", value: `${(confidence * 100).toFixed(1)}%`, inline: true },
      { name: "Right or wrong?", value: REACTION_LEGEND, inline: false },
    ],
    image: { url: "attachment://capture.jpg" },
    footer: { text: captureKey },
    timestamp: timestamp ?? new Date().toISOString(),
  };
  void visible; // the guess is carried by `label`; the title stays a question

  const messageId = await postAsBot(env, embed, "label request", frame);
  if (messageId === null) return false;
  await seedLabelReactions(env, messageId);
  return true;
}

/**
 * Post a one-shot test embed (no inference involved) so the Worker → Discord
 * path can be verified end to end.
 */
export async function notifyDiscordTest(env: Env): Promise<boolean> {
  const embed: DiscordEmbed = {
    title: "🏔️ Worker notification test",
    description:
      "Test from the mountain-inference Worker. If you see this in Discord, the Worker → Discord bot-token path is healthy.",
    color: COLOR_TEST,
    footer: { text: "is-the-mountain-out • Notification test" },
    timestamp: new Date().toISOString(),
  };
  return (await postAsBot(env, embed, "test notification")) !== null;
}

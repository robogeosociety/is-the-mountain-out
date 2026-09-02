# Notifications

The `mountain-inference` Cloudflare Worker posts to Discord whenever the answer
to "is the mountain out?" changes. Discord delivers to desktop and mobile with
no per-IP rate limit (which is why it replaced the old ntfy.sh path — anonymous
ntfy publishing returned HTTP 429 from Cloudflare's shared egress IPs).

**The notification is also the feedback surface.** Each one carries the frame it
announced and three seeded reactions; tapping one tells the model it was right
or wrong, and the labeler bot (`BOT.md`) records it as a training label against
that exact frame. A 👎 on a false "the mountain is out!" is the highest-value
label this project can collect — precision over recall is the stated priority.

## What fires

The channel does two different jobs, and confidence decides which one a tick gets.
A third job — saying so when the pipeline itself is broken — was added 2026-08-24.

| Kind | Trigger | Title | Color |
|---|---|---|---|
| **Alert** | confirmed change, confidence ≥ `ALERT_MIN_CONFIDENCE` | 🏔️ The mountain is out! / ☁️ The mountain is gone | green / gray |
| **Label request** | any tick below that threshold, at most once per `LABEL_COOLDOWN_HOURS` | 🤔 Is the mountain out? | amber |
| **Pipeline down** | `HEALTH_ALERT_AFTER_FAILURES` consecutive failed ticks, then at most once per `HEALTH_REALERT_HOURS` | 🚨 Inference pipeline is down | red |
| **Pipeline recovered** | first success after a `down` alert — never rate-limited | ✅ Inference pipeline recovered | green |

An **alert** must be trustworthy — you act on it. A **label request** must be
informative — you correct it, and the correction is a training row. Those want
opposite inputs, which is why one threshold *routes* between them rather than
merely gating one.

Three rules shape it (`worker/src/transition.ts`):

- **Both directions.** The channel is the live answer, so it has to say when the
  answer stops being yes as well as when it starts.
- **A change must survive two consecutive ticks** before it can post. Raw
  predictions flap — real history has *out* 16:46, *gone* 17:01, *out* 18:01,
  which compare-to-previous-tick announced three times.
- **A held change only alerts when the model is sure.** Below the threshold
  `pending` stays armed, so the alert is **delayed to the first confident tick,
  never dropped** — and meanwhile that frame is exactly what a label request is
  for. On 2026-07-29/30 this keeps the four alerts at 0.88–1.00 and suppresses
  the four at 0.42–0.71, one of which reversed 30 minutes later.

Confidence is **binary** — p(out) = full + partial, versus p(not out) — because
that is the question being asked. A 0.45 Full / 0.45 Partial split is a
confident *yes*, not a coin flip; gating on the top class score would mistake it
for one.

Bookkeeping lives in `notify-state.json` next to `state.json` (announced state,
armed pending change, last label-request stamp). If it is missing or unreadable
the Worker re-adopts current visibility **silently**, so a lost file is never a
false alert. The cooldown stamp is written before the post, so a failed send
costs one ask rather than unlocking a burst.

A label request is pointless without a frame to ask about, so unlike an alert it
is skipped entirely when the capture fails to persist.

The embed **attaches** the announced frame rather than linking `webcam_url` —
that URL is `…/webcam2_latest.jpg`, so a linked image would quietly become a
different picture later, and you must be labeling the frame you can see. The
footer is the frame's R2 capture key, which is the contract `bot/labeler.py`
parses. If the frame fails to persist, the post falls back to a prose footer and
is simply not labelable.

## Pipeline health (`worker/src/health.ts`)

The first two rows above describe the mountain. These two describe whether we can
see it at all.

Between **2026-08-07T07:15Z** and 2026-08-24 the `*/15` tick failed **1,637 times
in a row** — the UW webcam image 404'd upstream — and the channel said nothing.
The tick's error path appended the failure to `history.jsonl` and called
`console.error`, and that was all it did: a log nobody reads and an ndjson file
nobody opens. The only visible symptom was a site quietly showing a stale reading,
which is indistinguishable from a slow day on the mountain.

So a sustained outage is now an event, not an absence of events. Two gates, the
same shape as the alert/label gates:

- **Threshold.** Nothing is said until `HEALTH_ALERT_AFTER_FAILURES` ticks have
  failed consecutively (default 4 = one hour). `history.jsonl` holds 142 "container
  is not listening" and 37 NOAA read timeouts that all self-healed on the next
  tick; paging on those teaches you to ignore the channel, which is how a
  seventeen-day outage goes unnoticed.
- **Cooldown.** Within one outage, re-alert at most every `HEALTH_REALERT_HOURS`
  (default 6) — about four messages a day rather than ninety-six.

**Recovery is deliberately not rate-limited.** The first success after an alerted
outage posts immediately. An all-clear is cheap, and a channel still showing a
pipeline that has actually been healthy for hours is its own kind of wrong.

The down alert carries the last error verbatim. The seventeen-day outage was
diagnosable from a single line of it; nobody ever saw that line.

Health bookkeeping lives in `health-state.json` in the public bucket, next to
`notify-state.json`. It is written before the message is posted, so a crash
between the two costs one alert rather than unlocking a burst of them.

## Why a bot token and not a webhook

Notifications used to go out through `DISCORD_WEBHOOK_URL`, and **every reaction
on them was silently dropped**. A bot without the privileged Message Content
intent receives `embeds: []` for messages authored by anyone else — including a
webhook — so the labeler could never read the capture key out of the footer.

Posting with the labeler's own bot token makes each notification a message the
bot authored, which Discord always delivers in full. That fixes reaction
labeling without requesting a privileged intent, and lets the bot edit its
🏷️ Label acknowledgment onto the post.

## Secrets

Both are set via `wrangler secret put` (the live deploy path is wrangler, not
Terraform). They are the same values the bot reads from `cf.env`.

| Secret | Required | Purpose |
|---|---|---|
| `DISCORD_BOT_TOKEN` | yes | The labeler bot's token. The Worker posts and seeds reactions as that bot. |
| `DISCORD_CHANNEL_ID` | yes | The channel to post in — the same one the bot watches. |

A missing secret degrades to a logged skip: notifications simply don't send.

```bash
# from worker/ — same values as cf.env:
npx wrangler secret put DISCORD_BOT_TOKEN
npx wrangler secret put DISCORD_CHANNEL_ID
npx wrangler secret delete DISCORD_WEBHOOK_URL   # retired
npx wrangler deploy   # secrets only go live on the next deploy
```

> **Obsolete:** the former `NTFY_TOPIC` / `NTFY_TOKEN` secrets and the gitignored
> `ntfy.key` / `ntfy-token.key` files at the repo root are no longer used.
> The local-only `tools/local_notifier.py` stopgap (and its
> `com.tommydoerr.mountain-notifier` launchd job) has been **retired**.

## Test

After deploying the Worker, hit the test endpoint:

```bash
curl -X POST https://mountain-inference.<your-cf-subdomain>.workers.dev/notify-test
```

That sends a one-shot blue test embed (no inference involved) — useful for
verifying the Worker secrets + Discord path without waiting for a change. The
endpoint always returns `202` (the publish is queued via `waitUntil`); if nothing
arrives in Discord, tail the Worker for the real result:

```bash
cd worker && npx wrangler tail --format json
```

`Discord test notification failed: 401` means the bot token is wrong or
regenerated, `403`/`404` means the bot can't see or post in that channel, and
`DISCORD_BOT_TOKEN/DISCORD_CHANNEL_ID not set` means a secret is missing.

You can also post directly from the terminal, bypassing the Worker:

```bash
curl -sS -X POST "https://discord.com/api/v10/channels/$DISCORD_CHANNEL_ID/messages" -H "Authorization: Bot $DISCORD_BOT_TOKEN" -H "Content-Type: application/json" -d '{"embeds":[{"title":"test","description":"hello","color":3447003}]}'
```

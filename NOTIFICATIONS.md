# Notifications

The project uses a **Discord webhook** for real-time push notifications. Discord delivers to desktop and mobile (iOS/Android) via the Discord app, with no per-IP rate limit on incoming webhooks — which is why it replaced the old ntfy.sh path (anonymous ntfy publishing returned HTTP 429 from Cloudflare's shared egress IPs).

> Related but distinct: the **reaction-labeling bot** (`BOT.md`) is a separate
> Gateway app that posts hourly captures and turns reactions into training
> labels. The two are designed to share a channel: on each transition the
> Worker persists the announced frame to R2 and footers the notification with
> its capture key, so a 👍/⛅/👎 reaction on a notification is picked up by the
> bot as a label too — a 👎 on a false "the mountain is out!" corrects the
> model with the exact offending frame.

## Subscribe

1. In the Discord server/channel you want the alerts in: **Channel → Edit → Integrations → Webhooks → New Webhook**.
2. Copy the webhook URL. It looks like `https://discord.com/api/webhooks/<id>/<token>`. **The URL is the secret** — anyone who has it can post to the channel. Store it only as a Worker secret (below), never in a committed file.

## What fires

The `mountain-inference` Cloudflare Worker (`worker/src/index.ts`) posts one embed on the **Not Out → visible** transition (i.e. when the previous tick's `state.json` had `is_out: false` and the current tick predicts `Full` or `Partial`). The Discord formatting lives in `worker/src/discord-mountain-notify.ts`.

| Trigger | Title | Color | Fields |
|---|---|---|---|
| Not Out → Full or Partial | 🏔️ The mountain is out! | green | Confidence (combined visible-class %), Classification (Full/Partial) |

The embed also attaches the webcam snapshot (`webcam_url`) as its image and stamps the prediction's `timestamp_utc`.

## Secrets

The Worker holds a single Discord secret, set via `wrangler secret put` (the live deploy path is wrangler, not Terraform):

| Secret | Required | Purpose |
|---|---|---|
| `DISCORD_WEBHOOK_URL` | yes | The Discord channel webhook URL to POST embeds to. A missing secret degrades to a logged skip — notifications simply don't send. |

```bash
# from worker/ — paste the webhook URL at the prompt (or pipe it in):
npx wrangler secret put DISCORD_WEBHOOK_URL
npx wrangler deploy   # secrets only go live on the next deploy
```

> **Obsolete:** the former `NTFY_TOPIC` / `NTFY_TOKEN` secrets and the gitignored
> `ntfy.key` / `ntfy-token.key` files at the repo root are no longer used by the Worker.
> Delete the old secrets with `npx wrangler secret delete NTFY_TOPIC` (and `NTFY_TOKEN`)
> and remove the key files. The local-only `tools/local_notifier.py` stopgap (and its
> `com.tommydoerr.mountain-notifier` launchd job) has been **retired** — the Worker posts
> to Discord directly, so the IP-dodging local fallback is no longer needed.

## Test

After deploying the Worker, hit the test endpoint:

```bash
curl -X POST https://mountain-inference.<your-cf-subdomain>.workers.dev/notify-test
```

That sends a one-shot blue test embed (no inference involved) — useful for verifying the Worker secret + Discord path without waiting for a transition. The endpoint always returns `202` (the publish is queued via `waitUntil`); if nothing arrives in Discord, tail the Worker to see the real result: `cd worker && npx wrangler tail --format json` then re-fire — a `Discord test notification failed: 401/404` line means the webhook URL is wrong or revoked, and `DISCORD_WEBHOOK_URL not set` means the secret is missing.

You can also post directly from the terminal (bypassing the Worker entirely):

```bash
curl -sS -X POST "$DISCORD_WEBHOOK_URL" -H "Content-Type: application/json" \
  -d '{"embeds":[{"title":"test","description":"hello","color":3447003}]}'
```

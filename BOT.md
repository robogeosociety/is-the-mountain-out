# Discord reaction-labeling bot

A gateway bot (`bot/`, discord.py) that turns a Discord channel into a mobile
labeling surface for the classifier: it posts fresh webcam captures on an
hourly daylight schedule, and your reactions become training labels.

```
👍 = Full (1) · ⛅ = Partial (2) · 👎 = Not Out (0)
```

Skin-tone variants (👍🏻…👍🏿) count as their base emoji. Labels land in the same
`labels.yaml` the classifier UI and `just train` already share — so reacting on
your phone feeds the next LoRA batch run with zero extra plumbing. This is a
**bot** (Gateway websocket), not a webhook: reaction events only arrive over
the Gateway, which is why the Worker's webhook notifier can't do this job.

## How a label happens

1. On the hour (inside civil dawn→dusk at the webcam's coordinates), the bot
   runs the standard collector capture: webcam frame + paired METAR →
   `data/YYYYMMDD/HHMMSS_us_UTC/…` → uploaded to R2 under the usual key. The
   METAR pairing matters — labeled samples stay fully featured for the model's
   dual (image ⊕ weather) input.
2. It posts the frame to the configured channel with the model's live
   prediction (from the public `state.json`, when `[bot] state_url` is set)
   and seeds the three label reactions. The embed **footer carries the capture's
   storage key** — the only message↔capture state, so nothing breaks on restart.
3. You tap a reaction. The bot maps emoji → class, checks the allowlist, and
   union-merges `{capture_key: class}` into `labels.yaml` (R2 is the source of
   truth; the merge never deletes keys added by the classifier UI). Each label
   also appends a provenance event to `labels/discord-events.jsonl`.
4. Re-labeling is just reacting again — last reaction wins. Reactions added
   while the bot was offline are recovered by a startup sweep of the last
   `sweep_hours` of channel history (majority vote; ties are skipped).
5. **The Worker's "mountain is out!" notifications are labelable too.** On a
   visibility transition the Worker persists the announced frame (+ METAR) to
   R2 under a standard capture key and footers the notification with it
   (`worker/src/index.ts`). The bot treats any *bot-authored* embed whose
   footer parses as a capture key — and whose key actually exists in storage —
   exactly like its own posts. A 👎 on a false positive is the highest-value
   label this system can collect (precision is the stated priority).
6. Next `just train` folds the Discord labels into the oversampled batch run
   like any others.

## Setup (once)

1. **Create the Discord app** (bot-per-purpose — its own app, not a shared
   token; register it in `tommyroar/discobots` → `DISCORD.md`). In the
   [developer portal](https://discord.com/developers/applications): New
   Application → Bot → copy the token. No privileged intents needed (the bot
   only reads embeds of its own messages, so Message Content stays off).
2. **Invite it** with scope `bot` and permissions: View Channel, Send Messages,
   Embed Links, Attach Files, Add Reactions, Read Message History (permissions
   integer `117824`).
3. **Configure** `cf.env` (copy from `cf.env.example`): `DISCORD_BOT_TOKEN`,
   `DISCORD_CHANNEL_ID` (the labeling channel), optional
   `DISCORD_ALLOWED_USER_IDS` (comma-separated; empty = any human's reactions
   count — set it on shared servers), plus the existing R2 credentials.
4. **Tune** the optional `[bot]` section in `mountain.toml`
   (`post_interval_seconds`, `sweep_hours`, `state_url`, fixed
   `window_start`/`window_end` overriding the solar window).

Point `DISCORD_CHANNEL_ID` at the same channel the Worker's webhook
notifications post to (`#mountain`) — that's what makes notification reactions
labelable. Messages that aren't bot-authored embeds with a real capture-key
footer (older prose-footer notifications, human messages) are ignored.

## Run

```bash
just bot-post-once   # capture + post one labelable message, then exit (setup check)
just bot             # the real thing: hourly daylight posts + reaction labeling

# under Nomad on the always-on mini (sources cf.env itself):
nomad job run nomad/bot.hcl
nomad job status mountain-labeler
```

The bot needs the `bot` dependency group (`uv run --group bot …` — the just
recipes and Nomad job spell it): discord.py is deliberately not a main
dependency so the inference container image never installs it.

## Troubleshooting

- **`DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID must be set`** — cf.env not
  sourced (the Nomad job and `just` recipes run from the repo root, where
  `bot run` reads the environment; `set -a && source cf.env` first when
  running by hand).
- **Posts but reactions don't label** — check `DISCORD_ALLOWED_USER_IDS`
  (your user id must be listed if it's non-empty) and that the reaction is one
  of 👍/⛅/👎 (anything else is ignored by design).
- **No posts** — the bot only posts inside the daylight window; check
  `nomad alloc logs` for `capture post failed` (webcam/R2 hiccups skip the
  hour rather than posting an unlabelable message).
- **Missed reactions while down** — automatic: the startup sweep re-reads
  `sweep_hours` of history and records anything new.

# Discord reaction-labeling bot

> [!IMPORTANT]
> **This bot folds into the single `discobots` session** (fleet consolidation
> Phase 3). The mountain's reaction handler becomes a tool on the one
> `claude --channels` bot on the mini, alongside deploys and vault queries;
> the standalone `mountain-labeler` LaunchAgent and its own Discord app go
> away. `bot/labeler.py` is pure logic precisely so it can be lifted as-is.
>
> Until that lands, `bot/` still runs standalone — but note what already
> changed in Phase 2 (2026-09): **nothing posts from Cloudflare any more.** The
> mini's 15-minute tick appends alert and label-request records to
> `/Volumes/dev/mountain/live/announce.jsonl` (`[bot] announce_queue` in
> `mountain.toml`), and the bot's job is to drain that queue and post each one
> as itself — attaching the frame named by `capture_key`, footering that key,
> and seeding 👍⛅👎. The decision of *whether* to post is already made, by
> `bot/transition.py`, before the record is written.

A gateway bot (`bot/`, discord.py) that turns your reactions on the channel's
visibility notifications into training labels.

```
👍 = Full (1) · ⛅ = Partial (2) · 👎 = Not Out (0)
```

Skin-tone variants (👍🏻…👍🏿) count as their base emoji. Labels land in the same
`labels.yaml` the classifier UI and `just train` already share — so correcting a
call on your phone feeds the next LoRA batch run with zero extra plumbing. This
is a **bot** (Gateway websocket), not a webhook: reaction events only arrive
over the Gateway.

**The bot does not post on a schedule.** It used to publish an hourly capture to
label, which produced 24 near-identical "Not Out (100.0%)" quiz posts a day and
buried the notifications that actually meant something. The queued
state-change posts (`NOTIFICATIONS.md`) are now the whole labeling surface: they
fire when the answer changes, and the label you give is a correction on a
real prediction rather than a chore.

## How a label happens

1. The tick (`bot/transition.py`) either confirms a visibility change it is
   **confident** about (both directions, debounced over two inference ticks)
   or, when it is **unsure**, asks outright with a 🤔 label request — the frames
   a label actually teaches something. Either way the frame + paired METAR are
   already on the dev disk under the collector's standard capture key, which
   the queue record carries as `capture_key`; the bot posts it to the channel
   **as itself** — attaching the frame, footering its capture key, and seeding
   the three label reactions. The METAR pairing matters: labeled samples stay
   fully featured for the model's dual (image ⊕ weather) input.
2. You tap a reaction. The bot maps emoji → class, checks the allowlist, and
   union-merges `{capture_key: class}` into `labels.yaml` (the dev disk is the
   source of truth since 2026-09 — it was R2; the merge never deletes keys
   added by the classifier UI). Each label
   also appends a provenance event to `labels/discord-events.jsonl`, and the bot
   **edits a 🏷️ Label field onto the post** as acknowledgment — e.g. "New
   reaction (Not Out) recorded — 3/2004 training labels from Discord".
3. Re-labeling is just reacting again — last reaction wins. Reactions added
   while the bot was offline are recovered by a startup sweep of the last
   `sweep_hours` of channel history (majority vote; ties are skipped).
4. Next `just train` folds the Discord labels into the oversampled batch run
   like any others.

### Why the announcement is posted by the bot itself

The embed footer carries the capture key, and it is the **only** message↔capture
state — nothing on disk, so labeling survives a restart. Reading it requires the
bot to be able to see the embed.

A bot without the privileged **Message Content** intent gets `embeds: []` for
messages authored by anyone else. While notifications went through a webhook, the
footer came back empty and **every reaction on a notification was silently
dropped** — the feature was wired end to end and could never have worked.
Posting as the bot makes each notification a message the bot authored, which
Discord always delivers in full. No privileged intent needed, and the bot gains
the ability to edit its acknowledgment onto the post.

Messages that aren't bot-authored embeds with a real capture-key footer (older
prose-footer notifications, human messages) are ignored, and a parsed key is
additionally gated on actually existing in storage before anything is recorded.

## Setup (once)

1. **Create the Discord app** (bot-per-purpose — its own app, not a shared
   token; register it in `tommyroar/discobots` → `DISCORD.md`). In the
   [developer portal](https://discord.com/developers/applications): New
   Application → Bot → copy the token. No privileged intents needed.
2. **Invite it** with scope `bot` and permissions: View Channel, Send Messages,
   Embed Links, Attach Files, Add Reactions, Read Message History (permissions
   integer `117824`).
3. **Configure** `cf.env` (gitignored, create it by hand): `DISCORD_BOT_TOKEN`,
   `DISCORD_CHANNEL_ID` (the labeling channel), optional
   `DISCORD_ALLOWED_USER_IDS` (comma-separated; empty = any human's reactions
   count — set it on shared servers). Nothing else needs these values any more:
   the announcements come from this same process, off the local queue.
4. **Tune** the optional `[bot]` section in `mountain.toml` (`sweep_hours`,
   `state_url`, `announce_queue`).

`DISCORD_CHANNEL_ID` must be the same channel in both places — that's what makes
the notifications labelable.

## Run

```bash
just bot-post-once   # capture + post one labelable message, then exit (setup check)
just bot             # the real thing: reaction labeling + startup sweep

# under Nomad on the always-on mini (sources cf.env itself):
# Nomad is retired; run it directly (or under a KeepAlive LaunchAgent)
just bot
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
- **A notification has no reactions and a prose footer** — the announce record
  had no `capture_key` (the tick's capture failed), so the post is deliberately
  unlabelable rather than pointing at a missing frame.
  `~/Library/Logs/mountain-tick.err.log` on the mini shows the reason.
- **Reactions on a message posted before this change do nothing** — correct.
  Those are webhook-authored; the bot cannot read their embeds. Only posts from
  the bot token are labelable.
- **Missed reactions while down** — automatic: the startup sweep re-reads
  `sweep_hours` of history and records anything new.
- **Quiet channel** — expected. Posts now track real visibility changes
  (recently ~2–6 a day), not a fixed hourly cadence.

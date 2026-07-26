# DRAFT — the LoRA feedback loop on Cloudflare

> Status: **draft plan, deliberately parked** (Tommy, 2026-07-26: "leave cloudflare as a
> draft plan"). Nothing here is scheduled work. The live loop it would evolve —
> reaction labeling (BOT.md) + supervisor-scheduled retraining — runs on the mini
> today, and stays there until this is picked up.

## Why (when picked up)

The 2026-05 migration (`cloudflare-migration-spec.md`, mini checkout) already moved
storage (R2), inference (Worker + Container), the SPA (Pages), and notifications
(Worker webhook) to Cloudflare. The LoRA *feedback loop* is the remaining
mini-resident slice: reaction capture, event gating, training, and checkpoint
promotion. Most of it is orchestration, not computation — and orchestration is what
Cloudflare does well.

## What can move — and what can't

| Piece | Today | Cloudflare target |
| --- | --- | --- |
| Label store + provenance | R2 (`labels.yaml`, `labels/discord-events.jsonl`) | already there ✅ |
| Event gating ("new labels since last train?") | supervisor cron on the mini reads R2 | Worker cron reading the same objects — trivial port |
| Training trigger → mini | supervisor REGISTRY entry | the deploy-gate **dispatch lane** (supervisor#26 pattern): CF decides *when*, the mini only ever executes |
| Training compute (MPS) | mini, `uv run training batch` | **cannot move** — no GPU/Metal on Workers; stays on Mac hardware |
| Reaction capture (Gateway websocket) | `mountain-labeler` bot on the mini | **cannot be a plain Worker** — a Durable Object *can* hold an outbound websocket, but that's a discord.py rewrite; weigh against Discord's HTTP-interactions model (buttons instead of reactions) before porting |
| Eval + promotion gate | none (batch always promotes best-val) | Worker step: compare candidate metrics vs live, promote/rollback via R2 checkpoint keys |
| Run telemetry → #mountain | training body posts via bot token | Worker posts via the same webhook rail as notifications |

## Sketch (three small phases)

1. **Gate + trigger on CF:** a Worker cron owns "is there anything to train on";
   fires the mini through the dispatch lane. The mini-side body shrinks to "run
   `training batch`, upload candidate, exit."
2. **Promotion gate on CF:** candidate checkpoints land at `checkpoints/candidate/`;
   a Worker evaluates recorded metrics, promotes to `checkpoints/` (what the
   container cold-starts from) only on non-regression, posts the delta to #mountain.
3. **Reaction capture decision:** keep the mini gateway bot (cheap, working) or move
   to Worker + HTTP interactions with buttons — a UX change, decide deliberately.
   The taste-training plan (discobots#21, Phase T0) shares this exact decision.

## Non-goals

- No training compute on Cloudflare (no GPU; Workers AI can't fine-tune this model).
- No change to the labels.yaml contract — every surface keeps the union-merge.
- Not started until explicitly picked up; this doc is the parking spot.

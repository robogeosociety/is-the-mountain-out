# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Real-time image classifier that determines if Mount Rainier is visible ("out") using a live UW webcam feed. Uses a ConvNeXt Tiny backbone with LoRA fine-tuning, augmented with METAR weather data (visibility + ceiling) fed into the classifier head.

## Commands

### Python (uv)

```bash
# Tests
uv run pytest                              # all tests
uv run pytest train/tests/test_model.py -v # single test file

# Training
uv run training live        # continuous training loop (gradient accumulation)
uv run training once        # single capture + train cycle
uv run training batch data/20260222  # offline batch training on labeled dataset

# Data collection
uv run collect collect      # single capture (all webcams + METAR)
uv run collect live         # continuous collection loop

# Classification UI (FastAPI + Vite)
uv run classify start [data_folder]
uv run classify stop

# Discord reaction-labeling bot (see BOT.md; discord.py lives in the `bot` dependency group)
uv run --group bot bot run           # reaction labeling + startup sweep
uv run --group bot bot post-once     # one labelable post, then exit (setup check)

# Nomad job management
nomad job run nomad/collect.hcl          # start collector tray
nomad job run nomad/bot.hcl              # start Discord labeling bot
nomad job status mountain-collector      # check status
nomad alloc logs <alloc-id>              # view logs
nomad alloc logs -stderr <alloc-id>      # view error logs
```

### Frontend (ui/)

```bash
cd ui
npm run dev       # Vite dev server
npm run build     # type check + build
npm run lint      # ESLint
```

## Architecture

### Model (`train/model.py`)

`ConvNextLoRAModel` wraps `convnext_tiny` (timm) with PEFT LoRA adapters on the MLP `fc1`/`fc2` layers. The classifier head accepts a **dual input**: 768-dim image features concatenated with a 2-dim weather vector `[visibility, ceiling]` → Linear(770→256) + ReLU + Dropout → Linear(256→3). Three output classes: `0=Not Out`, `1=Full`, `2=Partial`.

Checkpoints saved to `train/checkpoints/`: `adapter_config.json`, `adapter_model.safetensors`, `classifier.pt`.

### Training Loop (`train/scheduler.py` + `train/utils.py`)

`WebcamStream` fetches JPEG from the webcam URL and converts directly to a `(1, 3, 224, 224)` tensor (no intermediate disk writes). `WeatherFetcher` queries METAR for KSEA and returns `[visibility_sm, ceiling_ft]`. The scheduler accumulates gradients over `N` captures before stepping (configurable in `mountain.toml`).

### Data Collection (`collect/collector.py`)

Writes timestamped directories: `data/YYYYMMDD/HHMMSS_us_UTC/{images/,metar/}`. Labels stored in `data/labels.yaml` as `{relative_path: label}`.

### Classification UI (`tools/classifier_server.py` + `ui/`)

FastAPI server writes its port to `data/classifier_server.port` at startup (dynamic port allocation). The React app (`ui/src/App.tsx`) polls `/api/images` for unlabeled batches (60 images), supports drag-to-select, hotkeys `1/2/0` for Full/Partial/None, and submits via `/api/label`. The Vite base path is `/classify/`; the API server reverse-proxies at that path.

### Discord labeling bot (`bot/`)

Gateway bot (discord.py, `bot` dependency group) that records 👍/⛅/👎 reactions as Full/Partial/Not-Out labels — the mobile counterpart to the classifier UI. **It does not post on a schedule**: the Worker's visibility-change notifications are the labeling surface, and the Worker writes them with *this bot's token* so they are bot-authored. That is load-bearing — without the privileged Message Content intent Discord blanks the embeds of any other author's messages, so while notifications came from a webhook the capture-key footer was unreadable and every reaction on one was silently dropped. `bot/labeler.py` is pure logic (emoji normalization, capture-key footers, union-merge into the shared `labels.yaml`); `bot/main.py` is the discord.py wiring (`on_raw_reaction_add`, startup sweep of missed reactions, and `post-once` as a manual setup check). See `BOT.md`.

### Configuration (`mountain.toml`)

Single source of truth for webcam URL, METAR station (`KSEA`), LoRA hyperparameters, checkpoint directory, collection intervals, and training schedule. Loaded via `train/config_loader.py`.

## Deployment (Cloudflare)

Inference runs as the `mountain-inference` Cloudflare Worker + Container (cron `*/15`), with R2 for storage and Pages for the SPA. There are two deploy paths; **prefer wrangler**:

- **wrangler (current, preferred):** `scripts/deploy-worker.sh` runs `npx wrangler deploy` from `worker/` and records a GitHub Deployment. This is what the live Worker uses (`last_deployed_from: wrangler`). Secrets are set out-of-band with `wrangler secret put` and only go live on the next `wrangler deploy`.
- **Terraform (`scripts/deploy-inference.sh` + `terraform/`):** retained as the intended path for reproducibility/IaC later, but the `terraform/` dir is currently absent and the script's TF path is stale — don't rely on it until it's rebuilt. Treat it as aspirational, not the working deploy.

Worker secrets (set via `wrangler secret put`):
- `DISCORD_BOT_TOKEN` / `DISCORD_CHANNEL_ID` — the labeler bot's credentials, so the Worker posts visibility changes *as that bot* and they're reaction-labelable. Same values as `cf.env`. See `NOTIFICATIONS.md`. (Replaced `DISCORD_WEBHOOK_URL`, which made posts unreadable to the bot; delete it with `npx wrangler secret delete DISCORD_WEBHOOK_URL`. The former `NTFY_TOPIC`/`NTFY_TOKEN` ntfy.sh secrets and the gitignored `ntfy.key`/`ntfy-token.key` files are also obsolete.)
- `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` ← `cf.env` — let the container pull its checkpoint from R2 on cold start.

Notifications fire on a visibility change in **both directions**, debounced over two consecutive inference ticks (`worker/src/transition.ts`, bookkeeping in `notify-state.json`; raw predictions flap). Each post attaches the announced frame — never a `webcam_url` link, which would silently become a different picture — and footers its R2 capture key so a reaction becomes a training label. Formatting and delivery: `worker/src/discord-mountain-notify.ts`. Failures are silent: `/notify-test` always returns `202` (publish is queued via `waitUntil`) and Discord errors are only `console.error`'d. To diagnose, `cd worker && npx wrangler tail --format json` and look for `Discord ... failed` or `DISCORD_BOT_TOKEN/DISCORD_CHANNEL_ID not set`. Worker tests: `cd worker && npm test`.

## Key Design Constraints

- **Zero-disk training:** Live frames go directly to tensors — never written to disk during live loops.
- **Dynamic port:** The classifier server picks a free port and writes it to `data/classifier_server.port`; the React UI fetches `config.json` at a relative path to discover it.
- **MPS device:** Apple Silicon (MPS) is the primary target; falls back to CPU.
- **Precision over recall:** The system is tuned to minimize false positives (announcing the mountain is out when it isn't).
## Pull requests — the "newspaper" framework

PR descriptions follow the **newspaper / information-pyramid** format: one self-contained
front page (kicker → headline → dek → masthead → why → what → mermaid flow → screens →
verification → risk) that reads top-to-bottom on an iPad-mini portrait display (1–2 pages;
up to 4 for very complex *code* changes). Rebuild from the **full** diff, never append.
Full rules: <https://github.com/robogeosociety/.github/blob/main/PR_FRAMEWORK.md>. CI validates
the body via the `pr-newspaper` workflow (the reusable gate in `robogeosociety/pr-newspaper`).

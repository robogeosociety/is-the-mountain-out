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

The `batch` command splits train/val **stratified per class, on unique labels, before oversampling, and seeded** (`--seed`, default 1337) — `stratified_split()`. That order is load-bearing: minority frames are oversampled ~5-7x, so splitting afterwards (as it did until 2026-07-30) put copies of the same image on both sides and inflated val accuracy. Val numbers from before that change are not comparable.

### Evaluation metrics (`train/metrics.py`)

**Accuracy is not the metric here.** The label set is 86.3% Not Out, so always answering "Not Out" scores 86.3%. Every validation pass builds a 3x3 confusion matrix and derives per-class precision/recall/F1/support, **macro-F1**, **balanced accuracy**, and a **Full+Partial "visible" binary view** — the last being the product question and what the Worker's alerts key on. Pure torch/stdlib arithmetic; scikit-learn stays a `dev`-group dependency. Results flow to `--json-summary` (`best_val_metrics`), each `per_epoch` record (`val_metrics`), `--progress-jsonl`, and the Discord embeds, which lead with macro-F1. All consumers tolerate the fields being absent (older runs). Details + the small-sample caveat: `TRAINING.md`.

### Data Collection (`collect/collector.py`)

Writes timestamped directories: `data/YYYYMMDD/HHMMSS_us_UTC/{images/,metar/}`. Labels stored in `data/labels.yaml` as `{relative_path: label}`.

### Classification UI (`tools/classifier_server.py` + `ui/`)

FastAPI server writes its port to `data/classifier_server.port` at startup (dynamic port allocation). The React app (`ui/src/App.tsx`) polls `/api/images` for unlabeled batches (60 images), supports drag-to-select, hotkeys `1/2/0` for Full/Partial/None, and submits via `/api/label`. The Vite base path is `/classify/`; the API server reverse-proxies at that path.

### Discord labeling bot (`bot/`)

Gateway bot (discord.py, `bot` dependency group) that records 👍/⛅/👎 reactions as Full/Partial/Not-Out labels — the mobile counterpart to the classifier UI. **It does not post on a schedule**: the Worker's visibility-change notifications are the labeling surface, and the Worker writes them with *this bot's token* so they are bot-authored. That is load-bearing — without the privileged Message Content intent Discord blanks the embeds of any other author's messages, so while notifications came from a webhook the capture-key footer was unreadable and every reaction on one was silently dropped. `bot/labeler.py` is pure logic (emoji normalization, capture-key footers, union-merge into the shared `labels.yaml`); `bot/main.py` is the discord.py wiring (`on_raw_reaction_add`, startup sweep of missed reactions, and `post-once` as a manual setup check). See `BOT.md`.

### Configuration (`mountain.toml`)

Single source of truth for webcam URL, METAR station (`KSEA`), LoRA hyperparameters, checkpoint directory, collection intervals, and training schedule. Loaded via `train/config_loader.py`.

## Deployment (Cloudflare)

Inference runs as the `mountain-inference` Cloudflare Worker + Container (cron `*/15`), with R2 for storage. The public site is a second Worker, **`is-the-mountain-out`** (`web/wrangler.toml`, `https://mountainisout.robogeosociety.xyz` — a Workers custom domain on the `robogeosociety.xyz` zone; the `is-the-mountain-out.tommy-b-doerr.workers.dev` hostname stays live alongside it), that serves the Vite build as static assets and answers `/state.json` + `/history.jsonl` same-origin from the R2 binding (`web/worker/index.ts`) — so the SPA has no cross-origin fetch and the bucket's CORS allowlist is out of the request path. That allowlist naming the pre-rename org is what blanked the site for weeks; see README → Outage post-mortem. There is no Cloudflare Pages project (the 2026-05-25 migration described one that was never created); the old GitHub Pages URL still serves a frozen 2026-05-25 build as a fallback.

**The site deploys from CI too — `.github/workflows/deploy-web.yml`**, on a push to `main` touching `web/**` (or `gh workflow run deploy-web.yml`): `web-ci.yml` (lint + `tsc -b` + vite build, the PR gate) then `npx wrangler deploy` in the `production-web` environment. By hand: `cd web && npm run deploy`. `vite dev` proxies `/state.json` to the bucket's r2.dev URL so the SPA code is identical in both.

**The site has its own CI credential — `CLOUDFLARE_API_TOKEN_WEB`**, vended from `cloudflare-tfvend` (`is_the_mountain_out.tf`, output `is_the_mountain_out_web_deploy`): `Workers Scripts: Edit` on the account plus `Workers Routes: Edit` scoped to the `robogeosociety.xyz` zone, because the custom domain in `web/wrangler.toml` is reconciled on every deploy via the zone's workers-routes API. It is deliberately not the inference Worker's `CLOUDFLARE_API_TOKEN` below: that token carries `Containers: Edit` and no zone grant, this one the reverse, so neither job's credential can do the other's work. Rotate or re-set it with `make -s output T=is_the_mountain_out_web_deploy | gh secret set CLOUDFLARE_API_TOKEN_WEB -R robogeosociety/is-the-mountain-out` on the Mac mini.

**The Worker deploys from CI — `.github/workflows/deploy-worker.yml`.** A push to `main` touching `worker/**` (or `gh workflow run deploy-worker.yml`) runs the worker tests + typecheck, then `npx wrangler deploy`, inside the `production` GitHub environment. Deploys are serialized (`cancel-in-progress: false`): one in flight is never cancelled. To require human approval, add a required reviewer to the `production` environment — no workflow change needed.

The GitHub Deployment record comes free with the job's `environment:` key: Actions itself opens a deployment and moves it `in_progress` → `success`/`failure`, with `environment_url` and a `log_url` to the run. That is *exactly* the bookkeeping `scripts/deploy-worker.sh` does by hand, so the workflow makes **no** explicit deployments-API calls — doing both would put two entries on the Environments page per deploy.

Deploy paths, in order of preference:

- **CI (normal):** the workflow above. Auth is the repo secret `CLOUDFLARE_API_TOKEN`.
- **`scripts/deploy-worker.sh` (break-glass):** same wrangler deploy + GitHub Deployment, run by a human under their own `wrangler login`. For when Actions is down, the token is expired, or an uncommitted tree must ship. It warns on a dirty tree, because the Deployment it records then points at a ref that does not match what went live.
- **Terraform (`scripts/deploy-inference.sh` + `terraform/`):** aspirational only — the `terraform/` dir does not exist and the script's TF path is stale. Do not rely on it.

The container image is *not* built or pushed by this workflow. `worker/wrangler.toml` pins an already-pushed tag in Cloudflare's managed registry; `.github/workflows/build-inference-image.yml` builds to GHCR and the `registry.cloudflare.com` push is still manual (`wrangler containers push`).

**CI credential — `CLOUDFLARE_API_TOKEN`.** The workspace rule is "auth via code flow, never mint tokens", but headless CI cannot run `wrangler login`'s browser flow, so a scoped API token is the sanctioned exception. Create it at *My Profile → API Tokens → Create Token*, scoped to account `d7adee58513c1b2f770ccaac90cf114f`, then:

```sh
gh secret set CLOUDFLARE_API_TOKEN -R robogeosociety/is-the-mountain-out
```

Required permissions — **two**, both **Account**-scoped and restricted to that one account:

| Permission | Why |
| --- | --- |
| `Workers Scripts: Edit` | Script upload, and with it the DO + R2 bindings, the `new_sqlite_classes` migration, and the `[triggers] crons` schedule. Non-negotiable. |
| `Containers: Edit` | `wrangler.toml` has a `[[containers]]` block, so every deploy also PATCHes the container application (`/accounts/{id}/containers/applications`) with the image ref, `max_instances`, `instance_type`. Without it the script uploads and the container step 403s. |

Deliberately **not** granted, each for a reason:

- `Workers R2 Storage: Edit` — an `[[r2_buckets]]` entry with an explicit `bucket_name` is pure script metadata; wrangler makes no R2 call. It only provisions buckets under the opt-in `--x-provision` flag. Add this only if CI ever runs `wrangler r2 …` itself (e.g. pushing a checkpoint).
- `Cloudflare Images: Edit` — a different product entirely (imagedelivery.net). Managed-registry auth is brokered through the *containers* API, and CI does not push images anyway.
- `User Details: Read` / `Memberships: Read` — only needed when wrangler has to discover the account. The workflow sets `CLOUDFLARE_ACCOUNT_ID`, so `/accounts` and `/memberships` are never called. Keep that env var: an account-owned token *cannot* carry User-scoped permissions (`/memberships` returns error 9106), so it is effectively mandatory there.
- `Workers Routes: Edit` (zone) — the inference Worker is `workers.dev` + cron only, no zone routes. The *site* Worker does need it, which is exactly why the site deploys with its own `CLOUDFLARE_API_TOKEN_WEB` (above) instead of this token.

Cloudflare's stock **"Edit Cloudflare Workers"** template is *not* sufficient on its own: it omits Containers. If an unexplained 403 appears, that template **plus `Containers: Edit`** is the low-risk superset. (Known upstream wrinkle: [workers-sdk#12483](https://github.com/cloudflare/workers-sdk/issues/12483) — `/containers/applications` 401 despite a valid containers scope.) The token is *only* for CI; the operator's laptop keeps using `wrangler login`.

Worker secrets themselves are still set out-of-band with `wrangler secret put` and only go live on the next deploy; CI does not manage them.

Worker secrets (set via `wrangler secret put`):
- `DISCORD_BOT_TOKEN` / `DISCORD_CHANNEL_ID` — the labeler bot's credentials, so the Worker posts visibility changes *as that bot* and they're reaction-labelable. Same values as `cf.env`. See `NOTIFICATIONS.md`. (Replaced `DISCORD_WEBHOOK_URL`, which made posts unreadable to the bot; delete it with `npx wrangler secret delete DISCORD_WEBHOOK_URL`. The former `NTFY_TOPIC`/`NTFY_TOKEN` ntfy.sh secrets and the gitignored `ntfy.key`/`ntfy-token.key` files are also obsolete.)
- `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` ← `cf.env` — let the container pull its checkpoint from R2 on cold start.

Confidence routes each tick to one of two posts (`worker/src/transition.ts`, bookkeeping in `notify-state.json`): an **alert** on a visibility change in both directions — debounced over two consecutive ticks and gated on binary confidence ≥ `ALERT_MIN_CONFIDENCE`, so an unsure change is delayed rather than dropped — or a **label request** (🤔 amber) when the model is unsure, rate-limited by `LABEL_COOLDOWN_HOURS`. Both are tunable in `worker/wrangler.toml` `[vars]`. Alerts must be trustworthy; label requests must be informative, which is why one threshold routes between them. Each post attaches the announced frame — never a `webcam_url` link, which would silently become a different picture — and footers its R2 capture key so a reaction becomes a training label. Formatting and delivery: `worker/src/discord-mountain-notify.ts`. Failures are silent: `/notify-test` always returns `202` (publish is queued via `waitUntil`) and Discord errors are only `console.error`'d. To diagnose, `cd worker && npx wrangler tail --format json` and look for `Discord ... failed` or `DISCORD_BOT_TOKEN/DISCORD_CHANNEL_ID not set`. Worker tests: `cd worker && npm test`.

## Key Design Constraints

- **Zero-disk training:** Live frames go directly to tensors — never written to disk during live loops.
- **Dynamic port:** The classifier server picks a free port and writes it to `data/classifier_server.port`; the React UI fetches `config.json` at a relative path to discover it.
- **MPS device:** Apple Silicon (MPS) is the primary target; falls back to CPU.
- **Precision over recall:** The system is tuned to minimize false positives (announcing the mountain is out when it isn't). Measured as `visible.precision` in `train/metrics.py` — before 2026-07-30 this constraint was stated but never measured.
## Pull requests — the "newspaper" framework

PR descriptions follow the **newspaper / information-pyramid** format: one self-contained
front page (kicker → headline → dek → masthead → why → what → mermaid flow → screens →
verification → risk) that reads top-to-bottom on an iPad-mini portrait display (1–2 pages;
up to 4 for very complex *code* changes). Rebuild from the **full** diff, never append.
Full rules: <https://github.com/robogeosociety/.github/blob/main/PR_FRAMEWORK.md>. CI validates
the body via the `pr-newspaper` workflow (the reusable gate in `robogeosociety/pr-newspaper`).

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
uv run collect capture-once # one headless capture (webcam + METAR); prints the key
uv run collect live         # continuous collection loop

# Classification UI (FastAPI + Vite)
uv run classify start [data_folder]
uv run classify stop

# Discord reaction-labeling bot (see BOT.md; discord.py lives in the `bot` dependency group)
uv run --group bot bot run           # reaction labeling + startup sweep
uv run --group bot bot post-once     # one labelable post, then exit (setup check)

# The production tick (what launchd runs on the mini every 15 minutes)
just tick                   # probe + capture + native inference + publish
just tick-local             # the same, without dispatching the Pages deploy
just publish                # gh workflow run publish.yml
just install-agent          # (re)install the LaunchAgent — GUI session only
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

**Accuracy is not the metric here.** The label set is 86.3% Not Out, so always answering "Not Out" scores 86.3%. Every validation pass builds a 3x3 confusion matrix and derives per-class precision/recall/F1/support, **macro-F1**, **balanced accuracy**, and a **Full+Partial "visible" binary view** — the last being the product question and what the channel's alerts key on
(`bot/transition.py`). Pure torch/stdlib arithmetic; scikit-learn stays a `dev`-group dependency. Results flow to `--json-summary` (`best_val_metrics`), each `per_epoch` record (`val_metrics`), `--progress-jsonl`, and the Discord embeds, which lead with macro-F1. All consumers tolerate the fields being absent (older runs). Details + the small-sample caveat: `TRAINING.md`.

### Data Collection (`collect/collector.py`)

Writes timestamped directories: `data/YYYYMMDD/HHMMSS_us_UTC/{images/,metar/}`. Labels stored in `data/labels.yaml` as `{relative_path: label}`.

### Classification UI (`tools/classifier_server.py` + `ui/`)

FastAPI server writes its port to `data/classifier_server.port` at startup (dynamic port allocation). The React app (`ui/src/App.tsx`) polls `/api/images` for unlabeled batches (60 images), supports drag-to-select, hotkeys `1/2/0` for Full/Partial/None, and submits via `/api/label`. The Vite base path is `/classify/`; the API server reverse-proxies at that path.

### Discord labeling bot (`bot/`)

Gateway bot (discord.py, `bot` dependency group) that records 👍/⛅/👎 reactions as Full/Partial/Not-Out labels — the mobile counterpart to the classifier UI. **It does not post on a schedule**: visibility-change notifications are the labeling surface. The mini's tick queues them to `/Volumes/dev/mountain/live/announce.jsonl` and the bot posts them *as itself* (until 2026-09 the Cloudflare Worker posted them using this bot's token, for the same reason). That is load-bearing — without the privileged Message Content intent Discord blanks the embeds of any other author's messages, so while notifications came from a webhook the capture-key footer was unreadable and every reaction on one was silently dropped. `bot/labeler.py` is pure logic (emoji normalization, capture-key footers, union-merge into the shared `labels.yaml`); `bot/main.py` is the discord.py wiring (`on_raw_reaction_add`, startup sweep of missed reactions, and `post-once` as a manual setup check). See `BOT.md`.

### Configuration (`mountain.toml`)

Single source of truth for webcam URL, METAR station (`KSEA`), LoRA hyperparameters, checkpoint directory, collection intervals, and training schedule. Loaded via `train/config_loader.py`.

## Deployment (Mac mini + GitHub Pages)

Since 2026-09 there is no Cloudflare in the live path — no Worker, no Container,
no R2, no Nomad, no `wrangler` in the tree. The previous arrangement (a `*/15`
cron Worker calling a Container, writing state to R2, a second Worker serving
the SPA) had been failing since 2026-09-15 and was the only thing forcing the
Workers Paid plan.

**The tick (`mini/`).** `com.robogeosociety.mountain-tick.plist` runs
`mini/tick.sh` every 900s on the mini. It (1) bounded-probes `/Volumes/dev` in a
child process and exits if the probe does not return within 5s, (2) runs
`collect capture-once`, (3) runs `tools/predict_state.py` natively on MPS with a
checkpoint from `/Volumes/dev/mountain/checkpoints`, writing
`live/state.json` (temp file + rename) and appending `live/history.jsonl`, and
(4) fires `gh workflow run publish.yml`. Read `mini/README.md` before touching
any of it.

Three things about that script are load-bearing:

- **The bounded probe.** `/Volumes/dev` has wedged twice; `access(2)` hangs
  while `ls` and `df` report a healthy mount. launchd will not start a new
  instance of a job whose previous one never exited, so an unbounded `stat`
  here does not fail the tick, it silences the site permanently. The probe runs
  in a child, the parent kills it at the deadline, and the parent never touches
  the path itself.
- **`install.sh` copies the script to `~/.local/libexec`.** launchd exec's the
  path in `ProgramArguments`; if that path were on the dev disk, a wedge would
  hang the exec before the script's own probe could run. Re-run `install.sh`
  after editing `tick.sh` — the agent runs the copy.
- **Discord is never called from the tick.** Announcements are *queued* to
  `live/announce.jsonl` (`--announce`), and the single Discord bot drains them.
  A chat API must not be able to stall a 15-minute job, and a bot restart must
  not lose an alert.

**The publish (`.github/workflows/publish.yml`).** `workflow_dispatch` (the
normal path, fired by the tick) plus a `*/15` schedule as a backstop.
`runs-on: [self-hosted, macOS, fleet]` because the job reads
`/Volumes/dev/mountain/live` off the runner host; `concurrency: pages`. It
builds `web/`, copies `state.json` + a bounded tail of `history.jsonl` into
`web/dist`, writes `CNAME` (`mountainisout.robogeosociety.xyz`) and
`.nojekyll` into the artifact, then `configure-pages` +
`upload-pages-artifact` + `deploy-pages`. It refuses to publish a missing or
unparseable `state.json` — a site stuck on "CHECKING…" looks like a front-end
bug and is not one.

The SPA fetches `` `${import.meta.env.BASE_URL}state.json` `` and vite's `base`
is `'./'`, so one artifact is correct both on the custom domain and on the
`robogeosociety.github.io/is-the-mountain-out/` fallback. Do not reintroduce an
absolute `/state.json`: it 404s on the project-pages path.

**Announcement policy** lives in `bot/transition.py` (a port of the Worker's
`transition.ts`, rules unchanged): a change must hold two consecutive ticks
*and* clear `--alert-min-confidence` (0.85 binary) to alert; an unsure tick
instead queues a 🤔 label request, rate-limited by `--label-cooldown-hours`.
`live/notify-state.json` carries the state machine between ticks. Tests:
`uv run pytest bot/tests/test_transition.py`.

**Migration leftovers, deliberately kept:** `collect/storage.py` still has the
R2 backends — `mini/r2-pull.sh` uses them to drain the two buckets onto the dev
disk once, and `[storage] backend` in `mountain.toml` can still be flipped back
to `"r2"`. Nothing in the live path does.

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

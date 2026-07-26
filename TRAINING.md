# LoRA training convention

This project shares a *harness* convention for LoRA training with the qwenbot/RAG
projects (`tommybot`). It is **not** a shared training library — the two trainers
have nothing in common at the tensor level (this repo trains a ConvNeXt-Tiny image
classifier with PyTorch + PEFT; tommybot trains a Qwen3-4B text LLM with MLX). What's
shared is the operational contract, so "kick off a LoRA training run" feels the same
across projects. The canonical write-up lives in tommybot's `TRAINING.md`; this is
the mountain-specific instance.

## The contract, here

1. **Training is an in-repo CLI command,** with a `just train` convenience recipe
   (`uv tool install rust-just` if you don't have `just`):

   ```sh
   just train data/labels.yaml 5
   # → uv run training batch --labels data/labels.yaml --epochs 5
   ```

   Labels arrive from two surfaces that share one `labels.yaml` (union-merged,
   R2 as source of truth): the bulk classifier UI (`uv run classify start`) and
   the Discord reaction-labeling bot (👍/⛅/👎 on hourly webcam posts — see
   `BOT.md`). A batch run picks both up with no extra flags.

   **Scheduled runs:** the robogeosociety/supervisor fires
   `scripts/scheduled-train.sh` (→ `python -m train.scheduled`) on the mini
   every Monday 04:00 — but only *trains* when `labels/discord-events.jsonl`
   has events newer than the R2 watermark (`labels/train-watermark.json`);
   idle weeks exit in seconds. Runs post start/finish telemetry to #mountain
   (best-val-loss delta included; a failed run keeps the watermark so next
   week retries). Epochs: `[training] scheduled_epochs`. Machine-readable
   results via `training batch --json-summary` — trust it over stdout, which
   historically claimed success on empty datasets and failed uploads.

   (the `training` console script → `train.scheduler:app`.)

2. **Training runs locally on the most capable machine, *not* under Nomad.** It
   prefetches the dataset from R2 and runs gradient descent on MPS — RAM-heavy and
   worth watching the per-epoch val loss for — so run it interactively on the best
   hardware available, not pinned to the weak always-on node. **Nomad here is
   reserved for the always-on collector service** (`collect.hcl`) plus the one-shot
   *capture* jobs (`once.hcl`, `capture_out.hcl`); training is a different shape and
   does not belong there.

3. **Adapters/checkpoints land where serving auto-discovers them:** the best
   checkpoint (by val loss) is written to `train/checkpoints/` (via
   `ConfigLoader.checkpoint_dir`) and uploaded to R2 for the inference container to
   pull on cold start. See `CHECKPOINTS.md` for model history.

4. **Weights and training data stay out of git** (see `.gitignore`). True in
   full since 2026-07-26 — the live `train/checkpoints/` weights were
   previously tracked, which dirtied the mini's checkout on every scheduled
   run and let the committed copy drift stale; R2 `checkpoints/` is the single
   source of truth and `load_checkpoint` falls back to it.

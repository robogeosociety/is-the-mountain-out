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
   the Discord reaction-labeling bot (👍/⛅/👎 on the Worker's alerts and
   label requests — see `BOT.md`). A batch run picks both up with no extra flags.

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

## What "good" means — read macro-F1, not accuracy

**Accuracy is not evidence on this label set.** Measured 2026-07-30 over 2010
labels: 1735 Not Out (86.3%) / 164 Partial (8.2%) / 111 Full (5.5%). A model that
answers "Not Out" to every frame scores **86.3% accuracy** while detecting the
mountain exactly never. A 99% headline is therefore mostly a report on how well
the model detects fog.

`train/metrics.py` computes, every validation pass, from one confusion matrix:

| Number | Why |
| --- | --- |
| **macro-F1** | Averaged over classes, so the majority class can't carry it. The all-"Not Out" predictor scores **0.31** here against 0.86 accuracy. This is the headline. |
| **balanced accuracy** | Mean per-class recall. Chance is 33.3%; the degenerate predictor scores exactly that. |
| per-class precision / recall / F1 / support | Where the errors actually are. `support` is printed everywhere on purpose — see the caveat below. |
| **visible (Full+Partial vs Not Out)** | The product question, and what the Worker's alerts key on. Precision leads: the repo's constraint is *precision over recall* — a false positive is an alert claiming the mountain is out when it isn't. |
| 3x3 confusion matrix | Rows = truth, columns = prediction. |

These land in `--json-summary` (`best_val_metrics`), in every `per_epoch` record
(`val_metrics`), and therefore in `--progress-jsonl`. Consumers must tolerate
their absence — summaries written before this existed have neither.

> **Caveat that matters more than any of the numbers.** With ~17 Full frames in
> val, **one flipped frame moves Full recall by ~6 percentage points.** Treat
> per-class recall on Full/Partial as a coarse signal with a ±1-frame error bar,
> not a precise measurement; a 3pt week-over-week "improvement" is noise. The
> fix is more Full/Partial labels, not more decimal places. The trainer prints a
> warning when any val class drops below 20 frames.

## Train/val split

Stratified per class, **on unique labels, before oversampling**, seeded
(`--seed`, default 1337). Each of those three properties fixes a real defect:

- **Before oversampling.** Minority frames are duplicated ~5-7x for class
  balance. The split previously ran on the *oversampled* list, so copies of the
  same image landed on both sides and validation scored the model on frames it
  had memorised. Historical val numbers — including the 99.8% that read as a
  success — are inflated by this and **are not comparable to numbers from this
  version**. Expect the first honest run to look worse. It isn't.
- **Stratified.** With 111 Full labels, a naive random split can leave single
  digits in val.
- **Seeded.** An unseeded split redrew the val set every week, so week-over-week
  deltas measured the dice as much as the model.

A class with a single example keeps it in train — spending it on val would make
the class untrainable *and* its recall a coin flip. Per-class val counts are
reported in the summary (`val_class_counts`) and in the Discord embed.

## Run telemetry

An unattended run reports to #mountain as it goes (`train/scheduled.py`), not
just at the end:

- **One start message**, edited in place with the final metrics when the run
  finishes — **macro-F1 (with a delta vs the previous run), balanced accuracy,
  per-class recall/precision, the visible-vs-not-out collapse, and the confusion
  matrix**, then best val loss, accuracy, dataset + val-split counts, duration
  breakdown, and **peak memory**. Accuracy is still there; it is no longer the
  headline, and its field name carries the 86.3% majority baseline next to it.
- **One message per saved checkpoint**, posted live. Only an *improvement*
  saves a checkpoint, so a 5-epoch run posts ~3: epoch, val loss with the delta
  over the previous best, macro-F1 / balanced accuracy, per-class recall, val
  accuracy, epoch time, memory, and how many of the 3 checkpoint files reached
  R2. Per-class recall is here because an improving val loss with a collapsing
  Full recall is a regression wearing a green badge.

The run-over-run macro-F1 delta comes from `best_macro_f1` on the R2 watermark
(`labels/train-watermark.json`), written alongside `best_val_loss`. The first run
after this landed has no previous value and says so.

The mechanism is `training batch --progress-jsonl PATH`: the trainer appends one
fsync'd JSON line per epoch (metrics + `memory_snapshot()` + whether a checkpoint
was saved), and the scheduler tails that file while the subprocess runs. Without
the flag the trainer writes nothing extra, so an interactive `just train` is
unchanged.

Memory probes are best-effort and platform-shaped — peak RSS via `getrusage`
(bytes on macOS, kilobytes on Linux; both handled), plus MPS allocated/driver or
CUDA allocated/peak when present. A probe that fails is omitted, never fatal.


# LoRA training convention

> [!CAUTION]
> **2026-09-24 — the camera changed, and the model did not survive it.**
> The UW ATG webcam died (404, removed upstream) and the pipeline moved to the
> **KING 5 Queen Anne tower camera** (`WEBCAMS.md`). Different location,
> different bearing, different focal length, different sky-to-city ratio, and
> Rainier is now a ~80x45 px smudge above the downtown towers instead of the
> UW framing the model was fit to.
>
> Concretely:
>
> - **The existing checkpoint is invalid for this framing.** It still loads and
>   still emits confident-looking probabilities; they mean nothing. Treat every
>   prediction as noise until a new checkpoint is trained on Queen Anne frames.
> - **Labels restart from zero.** The ~2,000 UW labels describe pixels that no
>   longer exist. New labels come in the same way as always — 👍/⛅/👎 on the
>   Discord posts — but from Queen Anne captures only (see "Labels at the
>   camera cut" below).
> - **Metrics are not comparable across the cut.** The 97.6% val accuracy and
>   the macro-F1 numbers below belong to the UW era. Do not compare a Queen
>   Anne run against them; the class balance, the difficulty and the input
>   distribution all changed at once. Start a new row in `CHECKPOINTS.md`.
> - **The camera can move.** Queen Anne is a PTZ broadcast camera on a
>   station's CDN, not a fixed instrument. If it is repointed, everything in
>   this box applies again. Re-verify the framing monthly — `WEBCAMS.md` has
>   the check.
>
> **Until a Queen Anne checkpoint exists, the site publishes no prediction at
> all.** This is enforced, not a convention: `[webcam] era` names the live
> camera, each checkpoint records its own era in `era.json` beside the weights
> (written on every save), and `tools/predict_state.py` refuses to load a
> checkpoint whose era does not match — it writes `status: "unvalidated"` with
> null `class_name`/`is_out`/`confidence`, the page shows CHECKING…, and the
> alert state machine is never consulted, so nothing is announced.
>
> **The labeling loop keeps running.** A frame is still queued to Discord on
> the ordinary label cooldown (`--label-cooldown-hours`, 4 h), carrying its
> capture key, so 👍/⛅/👎 reactions accumulate against real frames the whole
> time. That is the only route back to a working model.
>
> The gate clears itself: the first `just train` on Queen Anne labels stamps
> `era.json` with `king5-queenanne`; `just push-checkpoint` puts it on the
> mini, the eras match, and predictions resume on the next tick with no config
> change and no deploy.

## Training is Air-only

**The mini is 8 GB and must never train.** Decided 2026-07-04, restated in
`docs/consolidation-plan-2026-09.md` (2026-09-27) after the machine spent
fourteen days wedged: it has to hold an always-on Discord bot, the transit
tracker's WebSocket, the 15-minute inference tick and a GitHub runner in that
memory, and a ConvNeXt fine-tune with oversampled minority classes is the one
workload that reliably pushes it into swap. Inference is a forward pass every
900 seconds and costs it nothing. Training is a different shape and belongs on
the M4 Air, which is plugged in, has the RAM, and is the machine Tommy is
sitting at when he wants to watch a val loss move.

The split is therefore: **the mini writes labels, the Air writes checkpoints.**

```
mini (always on)                          Air (on demand)
────────────────                          ───────────────
Discord bot writes labels.yaml
captures accumulate in data/
                          ──►  just pull-labels   rsync over tommys-mac-mini.local
                                                  labels + only the referenced frames
                               just train         ConvNeXt + LoRA on MPS
                                                  best-val checkpoint, era.json stamped
                          ◄──  just push-checkpoint
tick loads the new weights                        rsync back + kickstart the daemon
status flips unvalidated → ok
```

### The loop, concretely

```sh
just pull-labels        # air/pull-labels.sh  — mini → Air
just train              # 5 epochs on ~/mountain/data/labels.yaml
just push-checkpoint    # air/push-checkpoint.sh — Air → mini, then kickstart
```

1. **Pull.** `air/pull-labels.sh` rsyncs `labels.yaml` and the `labels/`
   provenance log whole, then builds a manifest of exactly the captures those
   labels reference (plus each one's METAR) and pulls only those. The archive
   is thousands of frames; the labeled subset is the only part a run reads.
   `--all` mirrors everything if you want it.
2. **Train.** Unchanged — `training batch` on MPS. `just train` with no
   arguments resolves the labels path from `[paths] labels_dir`, so the
   command does not encode a machine.
3. **Push.** `air/push-checkpoint.sh` refuses to ship a checkpoint whose
   `era.json` does not match `[webcam] era` — a mis-stamped checkpoint does
   not produce a wrong answer, it produces a site stuck at CHECKING… with
   nothing in the logs explaining why, which is worse. The previous weights
   are kept in `checkpoints/.previous` on the mini. Then it kickstarts
   `system/com.robogeosociety.mountain-tick` so the next `state.json` is
   already the new model; without passwordless sudo on the mini it warns and
   lets the 15-minute interval do it.
4. **The gate clears itself.** `tools/predict_state.py` compares the loaded
   checkpoint's era to the live camera's on every tick. The first Queen Anne
   checkpoint makes them match, `status` goes from `"unvalidated"` to `"ok"`,
   the page stops saying CHECKING… and alerts resume. No config change, no
   deploy, no restart.

**Always `.local`, never the tailnet.** `[mini] host` is
`tommydoerr@tommys-mac-mini.local` and both scripts abort on a `*.ts.net`
address. Tailscale is not a dependency of this fleet; a training pull that
needs a re-auth nobody is watching is a pull that fails at the worst time.

### The label layout the Air expects

The mini is the only label **writer** — one Discord bot, reacting to 👍/⛅/👎
(`BOT.md`) — so this side is a one-way mirror. Do not hand-edit labels on the
Air; the next pull overwrites them.

| Path (relative to `[paths] labels_dir` / `captures_dir`) | Written by | Shape |
| --- | --- | --- |
| `labels.yaml` | bot, classifier UI | YAML map, `"<capture key>": 0\|1\|2` — 0 Not Out, 1 Full, 2 Partial. Union-merged on every write, never pruned |
| `labels/discord-events.jsonl` | bot | append-only provenance, one object per reaction: `timestamp`, `capture`, `label`, `class`, `user_id`, `source` |
| `labels/train-watermark.json` | scheduled runs | `best_val_loss`, `best_macro_f1`, and how far the events log has been consumed |
| `YYYYMMDD/HHMMSS_ffffff_UTC/images/<name>.jpg` | `collect capture-once` | the raw frame — **uncropped**; the burn-in strip comes off at load time |
| `YYYYMMDD/HHMMSS_ffffff_UTC/metar/metar.txt` | `collect capture-once` | the METAR row for that capture |

A **capture key** is the image path exactly as it appears above
(`20260927/101530_123456_UTC/images/frame.jpg`) — it is the key in
`labels.yaml`, the `capture_key` in `announce.jsonl`, and what
`pull-labels.sh` turns into an rsync manifest. `labels_dir` and `captures_dir`
are the same directory today, which is why `LocalStorage(data_root)` resolves
both; they are named separately because one is kilobytes pulled whole and the
other is gigabytes pulled selectively.

A label whose capture is missing is **not** an error: the trainer resolves
each key through the storage backend and `continue`s when the image or the
METAR is absent. That is what makes a selective pull safe, and what makes the
UW-era labels harmless rather than fatal.

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
   the mini as source of truth): the bulk classifier UI (`uv run classify start`) and
   the Discord reaction-labeling bot (👍/⛅/👎 on the tick's alerts and
   label requests — see `BOT.md`). A batch run picks both up with no extra flags.

   **Scheduled runs (historical):** the robogeosociety/supervisor fired
   `scripts/scheduled-train.sh` (→ `python -m train.scheduled`) on the mini
   every Monday 04:00. The supervisor's mini agents were retired in the
   2026-09 consolidation and training moved to the Air, so nothing runs this
   on a schedule today; the code and its telemetry are kept because a weekly
   run from the Air is the obvious next step. It only *trained* when `labels/discord-events.jsonl`
   has events newer than the R2 watermark (`labels/train-watermark.json`);
   idle weeks exit in seconds. Runs post start/finish telemetry to #mountain
   (best-val-loss delta included; a failed run keeps the watermark so next
   week retries). Epochs: `[training] scheduled_epochs`. Machine-readable
   results via `training batch --json-summary` — trust it over stdout, which
   historically claimed success on empty datasets and failed uploads.

   (the `training` console script → `train.scheduler:app`.)

2. **Training runs on the Air, on demand** — see *Training is Air-only* above.
   Gradient descent on MPS with oversampled minority classes is RAM-heavy and
   worth watching the per-epoch val loss for, so it runs interactively on the
   best hardware available and never on the weak always-on node. Nothing
   schedules it on the mini.

3. **Adapters/checkpoints land where serving auto-discovers them:** the best
   checkpoint (by val loss) is written to `train/checkpoints/` (via
   `ConfigLoader.checkpoint_dir`), and `just push-checkpoint` rsyncs it into
   the mini's `checkpoints/` — which the next 15-minute tick loads straight
   off the disk (it used to be uploaded to R2 for the inference container to
   pull on cold start). See `CHECKPOINTS.md` for model history.

4. **Weights and training data stay out of git** (see `.gitignore`). True in
   full since 2026-07-26 — the live `train/checkpoints/` weights were
   previously tracked, which dirtied the mini's checkout on every scheduled
   run and let the committed copy drift stale; R2 `checkpoints/` is the single
   source of truth and `load_checkpoint` falls back to it.

## Labels at the camera cut (2026-09-24)

The ~2,000 UW-era labels in `labels.yaml` are not wrong — they are about a
camera that no longer exists. They must not be mixed into a Queen Anne training
run, or the model learns the average of two unrelated views.

The trainer **skips samples whose image or METAR is missing** (it resolves each
key through the storage backend and `continue`s when either is absent), so a
label pointing at a deleted capture is harmless — it is silently dropped, not
an error. That means there is nothing to fix in this repo: `data/labels.yaml`
is gitignored and has never been tracked, and the live file lives on the mini,
under `[mini] root`.

What to do there, once, before the first Queen Anne training run:

```sh
cd ~/mountain/data                     # /Volumes/dev/mountain/data before the disk move
mv labels.yaml labels.uw-atg.yaml     # keep it: it is the UW era's record
: > labels.yaml                        # start empty; the bot appends to this
```

There is no need to delete or retag the UW **checkpoint**: the era gate already
refuses it (`train/checkpoint_era.py`), and keeping it costs nothing. Do not
hand-write an `era.json` naming the live camera to "unblock" the site — that
turns the gate off and puts UW-era guesses back on the page.

Keep the UW captures too — they cost little and they are the only evidence for
the pre-cut numbers in `CHECKPOINTS.md`. Just do not train on them.

**Roughly how many new labels before the numbers mean anything?** The UW model
was fit on ~2,000 labels with a 86/8/5 class split. The visible classes are the
scarce ones, and Rainier is smaller in this framing, so expect to need *at
least* a few hundred labels including a hundred-odd genuine "visible" frames —
which, given Seattle, means waiting out a season rather than an afternoon.

## What "good" means — read macro-F1, not accuracy

> [!NOTE]
> Every number in this section is from the **UW ATG era** (2026-02 → 2026-09).
> They are kept because the *method* is unchanged and the reasoning still
> applies, but they are not a baseline for the Queen Anne camera and a run
> against the new framing must not be compared to them.

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
| **visible (Full+Partial vs Not Out)** | The product question, and what the channel's alerts key on. Precision leads: the repo's constraint is *precision over recall* — a false positive is an alert claiming the mountain is out when it isn't. |
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

The run-over-run macro-F1 delta comes from `best_macro_f1` on the watermark
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


## Preprocessing: the burn-in crop

KING 5 burns a clock + branding strip into the bottom ~25 px of every frame. It
changes every frame and is the highest-contrast thing in the picture, so it is
removed **at load time** from the head of every transform pipeline —
`CropBurnIn` in `collect/frame.py`, driven by `[webcam] crop_bottom_px`:

| Path | Where the crop happens |
| --- | --- |
| batch training | `train_transform` / `val_transform` in `train/scheduler.py` |
| live training loop | `WebcamStream(..., crop_bottom_px=...)` in `train/utils.py` |
| inference tick | `tensor_from_bytes()` in `tools/predict_state.py` |
| capture/archive | **nowhere — captures are stored raw, on purpose** |

Cropping at load rather than at capture means the archive keeps the only
in-band record of when the camera says a frame was taken, and changing the
number later does not invalidate the captures.

The crop happens **before** `Resize(224)`. After resizing, 25 px of 1080 is
about 5 px of 224 — still enough white flickering text for a network to key on,
and by then it has been blended into the rows above it rather than removed.

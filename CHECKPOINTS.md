# Model Checkpoints & History

This document catalogs the major model checkpoints saved during the project's development and outlines expectations for future models.

> [!CAUTION]
> **Camera cut, 2026-09-24 — everything below is the UW ATG era.** The UW
> webcam died and the pipeline moved to the KING 5 Queen Anne tower camera
> (`WEBCAMS.md`). **Every checkpoint in this file is invalid for the new
> framing.** They load, they run, they emit confident probabilities — from a
> feature distribution that no longer exists. There is currently **no valid
> checkpoint for the live camera**, and the labels start from zero
> (`TRAINING.md` → *Labels at the camera cut*).
>
> Metrics across the cut are not comparable: view, focal length, target size,
> sky/city ratio and class balance all changed together. The first Queen Anne
> checkpoint opens a new section below and starts its own baseline — do not
> quote it against the 97.6% / macro-F1 numbers from the UW era.
>
> If the Queen Anne camera is ever repointed (it is a PTZ broadcast camera),
> this same caution applies again to whatever was trained before the move.

## Era 2: KING 5 Queen Anne (2026-09-24 →)

### 6. *(none yet)*
- **Status:** **No checkpoint trained on this camera.** The tick runs and the
  site publishes, but predictions are the old model's guesses on a view it was
  never fit to — treat them as noise. Collect Queen Anne labels via the
  Discord reaction labeler (`BOT.md`), then `just train`.
- **Baseline to beat:** none. The first run sets it. Report macro-F1, balanced
  accuracy and the visible-class precision (`train/metrics.py`) as usual.

## Era 1: UW ATG webcam 2 (2026-02 → 2026-09) — historical

All checkpoints below were trained on UW ATG webcam 2 frames. Kept for the
record and for the reasoning; none of them is a valid model for the live site.

### 1. `checkpoints_v1_binary`
- **Date:** March 11, 2026
- **Architecture:** 2-Class Binary (Not Out / Out)
- **Dataset:** Phase 1 (1,260 images)
- **Strategy:** Heavy oversampling of the "Out" class (1.4% representation).
- **Status:** Retired. This model was highly accurate on unambiguous days but suffered from false positives during "Partial" visibility, leading to the 3-class redesign.

### 2. `checkpoints_v2_pre_reclassify`
- **Date:** March 14, 2026
- **Architecture:** 2-Class Binary
- **Status:** Archived. This was the final state of the binary model weights immediately prior to the dataset reclassification into 3 classes.

### 3. `train/checkpoints` (Phase 2 Baseline)
- **Date:** March 14, 2026
- **Architecture:** 3-Class (Not Out, Full, Partial)
- **Dataset:** 1,319 images (Phase 1 re-labeled)
  - Not Out: 1,254 (95.1%)
  - Full: 8 (0.6%)
  - Partial: 57 (4.3%)
- **Strategy:** Fresh ConvNeXt weights fine-tuned with 78x oversampling on "Full" and 11x on "Partial".
- **Status:** Superseded by the scheduled-run era (below).

### 4. Dev-disk `checkpoints/` (scheduled-run era, 2026-07-26 → 2026-09-24)
- **Status:** last UW-era checkpoint, and the file the tick still loads today —
  invalid for the Queen Anne framing (see the caution at the top).
- **Weights left git on 2026-07-26.** The live checkpoint was the R2
  `checkpoints/` object set until 2026-09; it is now
  `/Volumes/dev/mountain/checkpoints/` on the mini (pulled out of R2 once by
  `mini/r2-pull.sh`, backed up nightly by restic), rewritten by the weekly
  supervisor-scheduled retrain (`TRAINING.md`) and loaded directly by the
  15-minute tick. `train/checkpoints/` is an untracked local working
  copy. Committing the
  binaries had two failure modes: the mini's checkout went permanently dirty
  after every scheduled run, and the committed copy silently drifted stale
  behind the model actually serving.
- **First scheduled run (2026-07-26):** 2,002 labels (1,729/109/164) incl. the
  first Discord reaction labels, 5 epochs — val_loss **0.0205**, val_acc
  **99.4%**, Full precision 0.88 → **0.98**. Run history from here lives in
  the #mountain training embeds and `labels/train-watermark.json`.

## Future Expectations

### The April 14 Fine-Tuned Model (`v3_spring`)
The current Phase 2 data collection run is scheduled to conclude on **April 14**. This 30-day "Diffuse Spring" solar plan captures 21 images a day with a focus on capturing rapid clearing events and diverse cloud layers.

Once the new dataset is labeled and the model is fine-tuned, we will run the new `tools/evaluate.py` script. 

**Expectations:**
1. **Partial Class Separation:** A massive improvement in F1 and Precision for the "Partial" class. The model should better understand the boundary between a fully obscured mountain and one peeking through a marine layer.
2. **False Positives:** A continued suppression of false positives (predicting Full/Partial when it is Not Out).
3. **METAR Reliance:** The model should demonstrate an even stronger fusion of visual and high-frequency METAR data to handle late-spring volatile weather.

## Evaluation
To compare a checkpoint against a labeled dataset, use the evaluation script:
```bash
uv run python tools/evaluate.py --checkpoint path/to/checkpoint --labels path/to/labels.yaml
```
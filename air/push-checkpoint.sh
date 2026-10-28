#!/bin/bash
# Ship a checkpoint trained on this Mac to the mini, and make the tick pick it
# up now instead of at the next quarter hour.
#
#   bash air/push-checkpoint.sh                    # the configured checkpoint_dir
#   bash air/push-checkpoint.sh train/checkpoints  # a specific run
#   bash air/push-checkpoint.sh --dry-run
#   bash air/push-checkpoint.sh --force            # push despite an era mismatch
#
# The other half of the Air-only training loop (see air/pull-labels.sh):
# pull labels -> `just train` on MPS -> push the weights back. The mini's
# 15-minute tick loads whatever is in its checkpoint directory, so this is the
# whole deploy: no image, no restart, no registry.
#
# The era check is the reason this is a script and not an rsync alias. Every
# save stamps era.json with the camera it was trained on, and
# tools/predict_state.py refuses a checkpoint whose era does not match
# [webcam] era — it publishes status "unvalidated" and no prediction at all.
# Pushing a mis-stamped or unstamped checkpoint therefore does not produce a
# wrong answer; it produces a site that stays at CHECKING… with nothing in the
# logs saying why. Catching it here, before the copy, is cheaper.
set -uo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=air/_config.sh
. "$SRC_DIR/_config.sh"

DRY_RUN=0
FORCE=0
LOCAL_CKPT=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --force) FORCE=1 ;;
        -h | --help)
            sed -n '2,25p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        -*) echo "unknown argument: $arg" >&2 && exit 2 ;;
        *) LOCAL_CKPT="$arg" ;;
    esac
done

mountain_load_config || exit 1
command -v rsync >/dev/null || mountain_die "rsync not on PATH"

[ -n "$LOCAL_CKPT" ] || LOCAL_CKPT="$AIR_CHECKPOINTS"
LOCAL_CKPT="${LOCAL_CKPT%/}"
[ -d "$LOCAL_CKPT" ] || mountain_die "no checkpoint directory at $LOCAL_CKPT"

# --- 1. is this a complete checkpoint? -----------------------------------
for required in adapter_config.json adapter_model.safetensors classifier.pt; do
    [ -f "$LOCAL_CKPT/$required" ] \
        || mountain_die "$LOCAL_CKPT/$required is missing — that is not a checkpoint"
done

# --- 2. is it for the camera the mini is actually watching? ---------------
ckpt_era="$(uv run --quiet python - "$LOCAL_CKPT" <<'PY'
import sys

from train.checkpoint_era import read_checkpoint_era

print(read_checkpoint_era(sys.argv[1]) or "")
PY
)" || mountain_die "could not read the checkpoint era"

if [ -z "$ckpt_era" ]; then
    msg="$LOCAL_CKPT has no era.json — the tick will treat it as the dead UW camera and refuse it"
elif [ -n "$CAMERA_ERA" ] && [ "$ckpt_era" != "$CAMERA_ERA" ]; then
    msg="era mismatch: checkpoint is '$ckpt_era', the live camera is '$CAMERA_ERA' — the tick will refuse it"
else
    msg=""
fi
if [ -n "$msg" ]; then
    if [ "$FORCE" = "1" ]; then
        mountain_log "WARN: $msg (--force)"
    else
        mountain_die "$msg. Retrain on current labels, or --force if you mean it."
    fi
fi
mountain_log "checkpoint era '${ckpt_era:-none}' vs camera '${CAMERA_ERA:-unset}'"

# --- 3. copy it over -----------------------------------------------------
RSYNC_FLAGS=(-a --human-readable --info=stats1)
[ "$DRY_RUN" = "1" ] && RSYNC_FLAGS+=(--dry-run)
# Keep the outgoing weights: a checkpoint is ~3 MB and "put yesterday's model
# back" should not require a retrain.
RSYNC_FLAGS+=(--backup --backup-dir="$MINI_CHECKPOINTS/.previous")

mountain_log "$LOCAL_CKPT -> $MINI_HOST:$MINI_CHECKPOINTS"
ssh "$MINI_HOST" "mkdir -p '$MINI_CHECKPOINTS'" \
    || mountain_die "cannot reach $MINI_HOST — is the mini awake on the LAN?"
rsync "${RSYNC_FLAGS[@]}" "$LOCAL_CKPT/" "$MINI_HOST:$MINI_CHECKPOINTS/" \
    || mountain_die "checkpoint push failed"

if [ "$DRY_RUN" = "1" ]; then
    mountain_log "dry run — not kickstarting $TICK_LABEL"
    exit 0
fi

# --- 4. run a tick now ---------------------------------------------------
#
# The tick is a system LaunchDaemon since 2026-09-27 (it has to survive
# `fdesetup authrestart`, which leaves no user session), so kickstarting it
# needs root. Best effort by design: without the sudoers line the next
# StartInterval fires within 15 minutes and loads the same weights.
if ssh "$MINI_HOST" "sudo -n launchctl kickstart -p system/$TICK_LABEL" 2>/dev/null; then
    mountain_log "kickstarted system/$TICK_LABEL — the next state.json is the new model"
else
    mountain_log "WARN: could not kickstart system/$TICK_LABEL (needs passwordless sudo"
    mountain_log "      for launchctl on the mini). The 15-minute tick will load it anyway."
fi

mountain_log "done — watch: ssh $MINI_HOST 'tail -n 20 ~/Library/Logs/mountain-tick.out.log'"

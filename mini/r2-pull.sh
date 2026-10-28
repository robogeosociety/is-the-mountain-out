#!/bin/bash
# One-shot: copy the two R2 buckets onto the dev disk, then never touch R2 again.
#
#   bash mini/r2-pull.sh              # both buckets
#   bash mini/r2-pull.sh --dry-run
#
# Run this ONCE, after the dev disk is back and before the tick LaunchAgent is
# installed. It lands:
#
#   is-the-mountain-out         → $MOUNTAIN_ROOT/data        (captures + METAR)
#     …/checkpoints/*           → $MOUNTAIN_ROOT/checkpoints (live weights)
#     …/labels.yaml             → $MOUNTAIN_ROOT/data/labels.yaml
#   is-the-mountain-out-public  → $MOUNTAIN_ROOT/live        (state.json, history.jsonl)
#
# Those paths are exactly what mini/tick.sh reads, so after this the mountain
# runs with Cloudflare unplugged. The buckets can then be deleted (Phase 5) —
# but verify the checkpoint loads first: `just predict` should print a class.
#
# Two transports, in order of preference:
#   1. rclone, if a remote for this R2 account is configured (fast, resumable)
#   2. boto3 via the repo's own collect.storage.R2Storage, using the R2 keys in
#      cf.env — no extra tooling, just slower
# `wrangler r2 object get` is deliberately NOT used: it fetches one key at a
# time and wrangler has no object-list command, so it cannot enumerate a bucket.
set -uo pipefail

DEV_ROOT="${MOUNTAIN_DEV_ROOT:-/Volumes/dev}"
MOUNTAIN_ROOT="${MOUNTAIN_ROOT:-$DEV_ROOT/mountain}"
PRIVATE_BUCKET="${MOUNTAIN_R2_BUCKET:-is-the-mountain-out}"
PUBLIC_BUCKET="${MOUNTAIN_R2_PUBLIC_BUCKET:-is-the-mountain-out-public}"
RCLONE_REMOTE="${MOUNTAIN_RCLONE_REMOTE:-r2}"
REPO_DIR="${MOUNTAIN_REPO:-$DEV_ROOT/is-the-mountain-out}"
ACCOUNT_ID="${R2_ACCOUNT_ID:-d7adee58513c1b2f770ccaac90cf114f}"
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

DATA_DIR="$MOUNTAIN_ROOT/data"
LIVE_DIR="$MOUNTAIN_ROOT/live"
CKPT_DIR="$MOUNTAIN_ROOT/checkpoints"

log() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }
die() { log "FATAL: $*"; exit 1; }

# Same bounded probe as the tick: never block on /Volumes.
( ls "$DEV_ROOT" >/dev/null 2>&1 ) & probe_pid=$!
waited=0
while kill -0 "$probe_pid" 2>/dev/null; do
    if [ "$waited" -ge 20 ]; then
        kill -9 "$probe_pid" 2>/dev/null
        die "$DEV_ROOT did not answer in 5s — the disk is wedged, not ready for this"
    fi
    sleep 0.25
    waited=$((waited + 1))
done
wait "$probe_pid" || die "$DEV_ROOT is not readable"

if [ "$DRY_RUN" = "1" ]; then
    log "DRY RUN — would populate $DATA_DIR, $CKPT_DIR, $LIVE_DIR"
fi
mkdir -p "$DATA_DIR" "$LIVE_DIR" "$CKPT_DIR" || die "cannot create $MOUNTAIN_ROOT"

# ---------- transport 1: rclone ----------
if command -v rclone >/dev/null && rclone listremotes 2>/dev/null | grep -qx "$RCLONE_REMOTE:"; then
    log "using rclone remote '$RCLONE_REMOTE:'"
    flags=(--progress --transfers 8 --checkers 16)
    [ "$DRY_RUN" = "1" ] && flags+=(--dry-run)

    rclone copy "$RCLONE_REMOTE:$PRIVATE_BUCKET" "$DATA_DIR" "${flags[@]}" \
        || die "rclone copy of $PRIVATE_BUCKET failed"
    rclone copy "$RCLONE_REMOTE:$PUBLIC_BUCKET" "$LIVE_DIR" "${flags[@]}" \
        || die "rclone copy of $PUBLIC_BUCKET failed"

    # checkpoints/ arrives inside the private bucket; hoist it so tick.sh's
    # --checkpoint-dir is a plain directory of weights.
    if [ -d "$DATA_DIR/checkpoints" ] && [ "$DRY_RUN" = "0" ]; then
        rsync -a "$DATA_DIR/checkpoints/" "$CKPT_DIR/" && rm -rf "$DATA_DIR/checkpoints"
        log "moved checkpoints → $CKPT_DIR"
    fi
    log "done. $(du -sh "$MOUNTAIN_ROOT" 2>/dev/null | cut -f1) on disk."
    exit 0
fi

log "no rclone remote '$RCLONE_REMOTE:' — falling back to boto3 via collect.storage"

# ---------- transport 2: boto3 through the repo's own R2Storage ----------
[ -d "$REPO_DIR" ] || die "repo not found at $REPO_DIR (set MOUNTAIN_REPO)"
cd "$REPO_DIR" || die "cannot cd to $REPO_DIR"

# R2 keys: environment first, then cf.env (gitignored, still on the mini).
if [ -z "${R2_ACCESS_KEY_ID:-}" ] && [ -f cf.env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./cf.env
    set +a
fi
[ -n "${R2_ACCESS_KEY_ID:-}" ] || die "no R2_ACCESS_KEY_ID (env or cf.env) and no rclone remote — cannot read R2"
export R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY

MOUNTAIN_PULL_DRY_RUN="$DRY_RUN" \
MOUNTAIN_PULL_ACCOUNT="$ACCOUNT_ID" \
MOUNTAIN_PULL_PRIVATE="$PRIVATE_BUCKET" \
MOUNTAIN_PULL_PUBLIC="$PUBLIC_BUCKET" \
MOUNTAIN_PULL_DATA="$DATA_DIR" \
MOUNTAIN_PULL_LIVE="$LIVE_DIR" \
MOUNTAIN_PULL_CKPT="$CKPT_DIR" \
uv run python - <<'PY' || die "boto3 pull failed"
"""Mirror both R2 buckets onto the dev disk using the repo's own client."""

import os
from pathlib import Path

from collect.storage import R2Storage

DRY = os.environ["MOUNTAIN_PULL_DRY_RUN"] == "1"
ACCOUNT = os.environ["MOUNTAIN_PULL_ACCOUNT"]


def mirror(bucket: str, dest: Path, checkpoint_dest: Path | None = None) -> None:
    store = R2Storage(account_id=ACCOUNT, bucket=bucket)
    keys = store.list_keys()
    print(f"{bucket}: {len(keys)} objects → {dest}")
    copied = skipped = 0
    for key in keys:
        # checkpoints/foo → <checkpoint_dest>/foo; everything else keeps its key
        if checkpoint_dest is not None and key.startswith("checkpoints/"):
            out = checkpoint_dest / key[len("checkpoints/") :]
        else:
            out = dest / key
        if out.exists():
            skipped += 1
            continue
        if DRY:
            copied += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(store.get(key))
        copied += 1
        if copied % 250 == 0:
            print(f"  {copied} copied…", flush=True)
    verb = "would copy" if DRY else "copied"
    print(f"{bucket}: {verb} {copied}, already present {skipped}")


mirror(
    os.environ["MOUNTAIN_PULL_PRIVATE"],
    Path(os.environ["MOUNTAIN_PULL_DATA"]),
    checkpoint_dest=Path(os.environ["MOUNTAIN_PULL_CKPT"]),
)
mirror(os.environ["MOUNTAIN_PULL_PUBLIC"], Path(os.environ["MOUNTAIN_PULL_LIVE"]))
PY

log "done. $(du -sh "$MOUNTAIN_ROOT" 2>/dev/null | cut -f1) on disk."

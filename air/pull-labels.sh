#!/bin/bash
# Pull the mini's labels — and only the captures those labels point at — onto
# this Mac, over the LAN.
#
#   bash air/pull-labels.sh              # labels + referenced captures
#   bash air/pull-labels.sh --dry-run
#   bash air/pull-labels.sh --all        # the whole capture archive (GBs)
#
# Training is Air-only (docs/consolidation-plan-2026-09.md, 2026-09-27): the
# mini has 8 GB and an always-on job to protect, so it collects and serves and
# never trains. The label WRITER stays there — the single Discord bot is the
# only thing that writes labels.yaml — which makes the mini the source of
# truth and this script a one-way mirror. Anything local under <labels_dir> is
# overwritten; do not hand-edit labels here.
#
# What moves (paths from [paths] and [mini] in mountain.toml):
#
#   <labels_dir>/labels.yaml                  capture key -> 0 Not Out | 1 Full | 2 Partial
#   <labels_dir>/labels/*.jsonl               provenance (discord-events, watermark)
#   <captures_dir>/YYYYMMDD/…/images/*.jpg    one frame per labeled key
#   <captures_dir>/YYYYMMDD/…/metar/metar.txt its weather row
#
# Selective on purpose: the archive is thousands of frames and the labeled
# subset is the only part a training run reads. Unlabeled captures are not
# worth a LAN copy, and re-running is cheap — rsync skips what is already
# here.
set -uo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=air/_config.sh
. "$SRC_DIR/_config.sh"

DRY_RUN=0
ALL_CAPTURES=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --all | --all-captures) ALL_CAPTURES=1 ;;
        -h | --help)
            sed -n '2,30p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *) echo "unknown argument: $arg" >&2 && exit 2 ;;
    esac
done

mountain_load_config || exit 1

command -v rsync >/dev/null || mountain_die "rsync not on PATH"

RSYNC_FLAGS=(-a --human-readable)
[ "$DRY_RUN" = "1" ] && RSYNC_FLAGS+=(--dry-run)

mountain_log "mini=$MINI_HOST labels=$MINI_LABELS -> $AIR_LABELS"
mkdir -p "$AIR_LABELS" "$AIR_CAPTURES" || mountain_die "cannot create the local tree"

# --- 1. the labels themselves (kilobytes; always pulled whole) ------------
rsync "${RSYNC_FLAGS[@]}" \
    "$MINI_HOST:$MINI_LABELS/labels.yaml" "$AIR_LABELS/labels.yaml" \
    || mountain_die "could not pull labels.yaml — is the mini awake on the LAN?"

# labels/ is the append-only provenance log plus the scheduled-run watermark.
# Absent on a fresh mini, which is not an error.
rsync "${RSYNC_FLAGS[@]}" \
    "$MINI_HOST:$MINI_LABELS/labels/" "$AIR_LABELS/labels/" 2>/dev/null \
    || mountain_log "no labels/ on the mini yet — provenance log skipped"

label_count="$(uv run --quiet python - "$AIR_LABELS/labels.yaml" <<'PY'
import sys

import yaml

with open(sys.argv[1]) as fh:
    print(len(yaml.safe_load(fh) or {}))
PY
)" || mountain_die "labels.yaml did not parse"
mountain_log "$label_count labels on disk"

# --- 2. the captures those labels point at --------------------------------
if [ "$ALL_CAPTURES" = "1" ]; then
    mountain_log "--all: mirroring the whole capture archive"
    rsync "${RSYNC_FLAGS[@]}" --exclude 'labels*' \
        "$MINI_HOST:$MINI_CAPTURES/" "$AIR_CAPTURES/" \
        || mountain_die "capture mirror failed"
    mountain_log "done"
    exit 0
fi

# A capture key is YYYYMMDD/HHMMSS_ffffff_UTC/images/<name>.jpg; the METAR for
# the same capture sits at …/metar/metar.txt. Both are fed to rsync through
# --files-from, which is one connection for the lot instead of one per frame.
manifest="$(mktemp -t mountain-pull-labels)" || mountain_die "mktemp failed"
trap 'rm -f "$manifest"' EXIT

uv run --quiet python - "$AIR_LABELS/labels.yaml" "$manifest" <<'PY' || mountain_die "could not build the capture manifest"
import sys
from pathlib import PurePosixPath

import yaml

labels_path, out_path = sys.argv[1], sys.argv[2]
with open(labels_path) as fh:
    labels = yaml.safe_load(fh) or {}

wanted: list[str] = []
seen: set[str] = set()
for key in labels:
    key = str(key).strip().lstrip("/")
    if not key or key in seen:
        continue
    seen.add(key)
    wanted.append(key)
    # …/images/frame.jpg -> …/metar/metar.txt. A label whose key is not a
    # capture path (there should be none) just contributes itself.
    parts = PurePosixPath(key).parts
    if len(parts) >= 3 and parts[-2] == "images":
        wanted.append(str(PurePosixPath(*parts[:-2]) / "metar" / "metar.txt"))

with open(out_path, "w") as fh:
    fh.write("\n".join(wanted) + ("\n" if wanted else ""))
print(f"{len(seen)} labeled captures ({len(wanted)} files) in the manifest")
PY

if [ ! -s "$manifest" ]; then
    mountain_log "no labeled captures yet — nothing to pull"
    exit 0
fi

# --ignore-missing-args: a label can outlive its capture (the trainer skips
# those samples deliberately), and one pruned frame must not fail the pull.
rsync "${RSYNC_FLAGS[@]}" --ignore-missing-args --info=stats1 \
    --files-from="$manifest" \
    "$MINI_HOST:$MINI_CAPTURES/" "$AIR_CAPTURES/" \
    || mountain_die "capture pull failed"

mountain_log "done — train with: just train"

#!/bin/bash
# One mountain tick: capture a frame, run inference natively, publish.
#
# Driven by mini/launchd/com.robogeosociety.mountain-tick.plist (a system
# LaunchDaemon, StartInterval 900) on the Mac mini. This is the entire
# production path since 2026-09 — there is no Cloudflare Worker, no container,
# and no cron in the cloud. The tick:
#
#   1. bounded-probes the dev disk and BAILS if it is wedged (only when
#      something it needs still lives there)
#   2. `collect capture-once`      — webcam frame + METAR to $MOUNTAIN_ROOT/data
#   3. `tools/predict_state.py`    — native PyTorch (MPS), local checkpoint
#      → live/state.json, live/history.jsonl, live/announce.jsonl
#   4. `gh workflow run publish.yml` — GitHub Pages picks the files up
#
# Nothing here talks to Discord. Announcements are queued to announce.jsonl and
# the single Discord bot drains them; a tick must never block on a chat API.
#
# Run it by hand exactly the way launchd does:  bash mini/tick.sh
set -uo pipefail

# launchd hands a job a minimal PATH — a DAEMON gets even less than an agent,
# and no login shell ever runs to fix it. Everything below (uv, gh, git) lives
# in ~/.local/bin on this fleet, rsync and friends in /opt/homebrew/bin; the
# rest is the system default. The plist sets the same string, so a by-hand run
# and the daemon resolve the same binaries.
export PATH="$HOME/.local/bin:$HOME/.pixi/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"

DEV_ROOT="${MOUNTAIN_DEV_ROOT:-/Volumes/dev}"
# The data root lives on the BOOT volume now (consolidation plan, 2026-09-27).
# It was $DEV_ROOT/mountain until the migration — the external disk that has
# wedged three times and takes the tick down with it whenever it does.
MOUNTAIN_ROOT="${MOUNTAIN_ROOT:-$HOME/mountain}"
REPO_DIR="${MOUNTAIN_REPO:-$DEV_ROOT/is-the-mountain-out}"
LIVE_DIR="${MOUNTAIN_LIVE:-$MOUNTAIN_ROOT/live}"
DATA_ROOT="${MOUNTAIN_DATA:-$MOUNTAIN_ROOT/data}"
CHECKPOINT_DIR="${MOUNTAIN_CHECKPOINTS:-$MOUNTAIN_ROOT/checkpoints}"
DEVICE="${MOUNTAIN_DEVICE:-mps}"
REPO_SLUG="${MOUNTAIN_REPO_SLUG:-robogeosociety/is-the-mountain-out}"
PROBE_TIMEOUT_SECONDS="${MOUNTAIN_PROBE_TIMEOUT:-5}"
# Lock lives on the boot disk on purpose: a lock under /Volumes would be the
# very thing the probe exists to avoid touching. A daemon does not get the
# per-user $TMPDIR a login shell does, so the plist pins MOUNTAIN_LOCK_DIR to
# /tmp — otherwise a by-hand run and the daemon would take two different locks
# and cheerfully run two ticks at once.
LOCK_DIR="${MOUNTAIN_LOCK_DIR:-${TMPDIR:-/tmp}/mountain-tick.lock}"

log() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }
die() { log "FATAL: $*"; exit 1; }

# --- 1. bounded dev-disk probe -------------------------------------------
#
# /Volumes/dev has wedged twice (2026-07, 2026-08): access(2) hangs forever
# while `ls` and `df` report a healthy mount. An unbounded stat here does not
# fail the tick, it hangs it — and launchd will not re-run a job whose previous
# instance never exited, so one hung probe silences the site indefinitely. So
# the probe runs in a CHILD, and the parent kills it at the deadline and exits
# non-zero. The parent never touches the path itself.
probe_dev_disk() {
    local target="$1" deadline="$2" pid waited=0
    ( ls "$target" >/dev/null 2>&1 && test -r "$target" ) &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        if [ "$waited" -ge "$((deadline * 4))" ]; then
            kill -9 "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
            return 2   # wedged: the probe never returned
        fi
        sleep 0.25
        waited=$((waited + 1))
    done
    wait "$pid"        # 0 = readable, non-zero = absent/unreadable
}

# The probe is only worth running while something the tick needs still lives
# on the dev disk. Since the data root moved to the boot volume the repo is
# usually the last holdout; once it moves too this becomes a no-op and the
# tick stops caring whether /Volumes/dev is mounted at all. Keeping the code
# rather than deleting it is deliberate: the disk is still attached, and a
# path can drift back under it through one environment variable.
needs_dev_disk=0
case "$MOUNTAIN_ROOT/" in "$DEV_ROOT"/*) needs_dev_disk=1 ;; esac
case "$REPO_DIR/" in "$DEV_ROOT"/*) needs_dev_disk=1 ;; esac
case "$LIVE_DIR/" in "$DEV_ROOT"/*) needs_dev_disk=1 ;; esac
case "$DATA_ROOT/" in "$DEV_ROOT"/*) needs_dev_disk=1 ;; esac
case "$CHECKPOINT_DIR/" in "$DEV_ROOT"/*) needs_dev_disk=1 ;; esac

if [ "$needs_dev_disk" = "1" ]; then
    probe_dev_disk "$DEV_ROOT" "$PROBE_TIMEOUT_SECONDS"
    probe_status=$?
    case "$probe_status" in
        0) ;;
        2) die "dev disk probe on $DEV_ROOT did not return within ${PROBE_TIMEOUT_SECONDS}s — disk is wedged, skipping this tick" ;;
        *) die "dev disk $DEV_ROOT is not readable (probe exit $probe_status) — skipping this tick" ;;
    esac
fi

# --- 2. single instance ---------------------------------------------------
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    lock_pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || echo '')"
    if [ -n "$lock_pid" ] && kill -0 "$lock_pid" 2>/dev/null; then
        log "another tick (pid $lock_pid) is still running — skipping"
        exit 0
    fi
    log "clearing a stale lock from pid ${lock_pid:-unknown}"
    rm -rf "$LOCK_DIR"
    mkdir "$LOCK_DIR" || die "could not take the tick lock at $LOCK_DIR"
fi
echo "$$" > "$LOCK_DIR/pid"
trap 'rm -rf "$LOCK_DIR"' EXIT

[ -d "$REPO_DIR" ] || die "repo not found at $REPO_DIR (set MOUNTAIN_REPO)"
cd "$REPO_DIR" || die "cannot cd to $REPO_DIR"
mkdir -p "$LIVE_DIR" "$DATA_ROOT" || die "cannot create $LIVE_DIR / $DATA_ROOT"
command -v uv >/dev/null || die "uv not on PATH ($PATH)"

log "tick start — repo=$REPO_DIR live=$LIVE_DIR device=$DEVICE"

# --- 3. collect one frame + METAR ----------------------------------------
capture_key=""
if capture_key="$(uv run collect capture-once \
        --config mountain.toml \
        --data-root "$DATA_ROOT" \
        --print-key 2>>"${MOUNTAIN_COLLECT_LOG:-/dev/stderr}")"; then
    log "captured $capture_key"
else
    # A missed frame is not a reason to skip inference: predict_state fetches
    # the webcam itself, and a stale capture archive is cheaper than a stale
    # site. It IS a reason to leave capture_key empty, so nothing claims the
    # announced frame was archived when it was not.
    capture_key=""
    log "WARN: capture-once failed — continuing to inference without an archived frame"
fi

# --- 4. inference, natively ----------------------------------------------
predict_args=(
    --config mountain.toml
    --out "$LIVE_DIR/state.json"
    --log "$LIVE_DIR/history.jsonl"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --device "$DEVICE"
    --announce "$LIVE_DIR/announce.jsonl"
    --notify-state "$LIVE_DIR/notify-state.json"
)
[ -n "$capture_key" ] && predict_args+=(--capture-key "$capture_key")

if ! uv run python tools/predict_state.py "${predict_args[@]}" >/dev/null; then
    # predict_state already appended the failure to history.jsonl, so the site
    # can show staleness rather than a wrong answer. Publishing an unchanged
    # state.json buys nothing, so stop here.
    die "inference failed — see $LIVE_DIR/history.jsonl"
fi
# Report the status, not just the class: "unvalidated" (no checkpoint for this
# camera) and "stale" (frozen feed) are successful ticks that deliberately
# publish no prediction, and a log line saying "None" would read like a fault.
log "inference ok — $(/usr/bin/python3 -c 'import json,sys;s=json.load(open(sys.argv[1]));print(s.get("status","ok"), s.get("class_name") or "-", s["timestamp_utc"])' "$LIVE_DIR/state.json" 2>/dev/null || echo 'state.json written')"

# --- 5. publish -----------------------------------------------------------
if [ "${MOUNTAIN_SKIP_PUBLISH:-0}" = "1" ]; then
    log "MOUNTAIN_SKIP_PUBLISH=1 — not triggering publish.yml"
    exit 0
fi
if ! command -v gh >/dev/null; then
    log "WARN: gh not on PATH — publish not triggered (the workflow's */15 schedule will catch up)"
    exit 0
fi
if gh workflow run publish.yml -R "$REPO_SLUG" >/dev/null 2>&1; then
    log "publish.yml dispatched"
else
    # Best effort by design: publish.yml also runs on its own */15 schedule, so
    # a dispatch failure delays the site by minutes, it does not break it.
    log "WARN: gh workflow run publish.yml failed — falling back to the workflow schedule"
fi

log "tick done"

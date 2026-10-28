#!/bin/bash
# Shared config reader for the Air-side training scripts. Source it, do not
# run it:
#
#   . "$(dirname "$0")/_config.sh"
#   mountain_load_config          # exports AIR_* / MINI_* / TICK_LABEL
#
# One `uv run` resolves the whole of [paths] and [mini] from mountain.toml
# through train.config_loader, so the shell and the trainer can never disagree
# about where the labels are. Everything is shell-quoted before eval.

mountain_load_config() {
    local config="${MOUNTAIN_CONFIG:-mountain.toml}"
    local assignments
    if ! assignments="$(
        MOUNTAIN_CONFIG_PATH="$config" uv run --quiet python - <<'PY'
import os
import shlex

from train.config_loader import ConfigLoader

cfg = ConfigLoader(os.environ["MOUNTAIN_CONFIG_PATH"])
local, mini = cfg.paths, cfg.mini
out = {
    "AIR_CAPTURES": local["captures_dir"],
    "AIR_LABELS": local["labels_dir"],
    "AIR_CHECKPOINTS": local["checkpoint_dir"],
    "MINI_HOST": mini["host"],
    "MINI_ROOT": mini["root"],
    "MINI_CAPTURES": mini["captures_dir"],
    "MINI_LABELS": mini["labels_dir"],
    "MINI_CHECKPOINTS": mini["checkpoint_dir"],
    "TICK_LABEL": mini["tick_label"],
    "CAMERA_ERA": cfg.camera_era or "",
}
for key, value in out.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
    )"; then
        echo "FATAL: could not read $config (run from the repo root)" >&2
        return 1
    fi
    eval "$assignments"
    export AIR_CAPTURES AIR_LABELS AIR_CHECKPOINTS
    export MINI_HOST MINI_ROOT MINI_CAPTURES MINI_LABELS MINI_CHECKPOINTS
    export TICK_LABEL CAMERA_ERA

    # Tailscale is not a dependency of this fleet (2026-09-22). On the LAN the
    # mini answers to .local; a tailnet name here would make every training
    # pull depend on a re-auth nobody is watching.
    case "$MINI_HOST" in
        *.ts.net | *.ts.net:*)
            echo "FATAL: [mini] host is a tailnet name ($MINI_HOST)." >&2
            echo "       Use tommys-mac-mini.local — Tailscale is not a dependency." >&2
            return 1
            ;;
    esac
}

mountain_log() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }
mountain_die() {
    mountain_log "FATAL: $*"
    exit 1
}

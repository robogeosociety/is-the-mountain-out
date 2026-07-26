#!/bin/zsh
# Supervisor job body: weekly gated LoRA retrain + #mountain telemetry.
# Registered as `mountain-weekly-train` in robogeosociety/supervisor
# (supervisor/models.py) — Mon 04:00, heavy, timeout 2h. The gate lives in
# `python -m train.scheduled`: no new Discord label events → exit 0 in seconds.
#
# MOUNTAIN_REPO is overridable for the boot-disk escape hatch (the external
# /Volumes/dev has wedged before — see the supervisor registry's _REPO note).
set -euo pipefail

REPO="${MOUNTAIN_REPO:-/Volumes/dev/is-the-mountain-out}"
cd "$REPO"
set -a; source cf.env; set +a
export PATH="$HOME/.local/bin:$PATH"  # uv — the supervisor's child env is minimal

exec uv run python -m train.scheduled "$@"

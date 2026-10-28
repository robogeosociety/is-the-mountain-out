# is-the-mountain-out tasks.
#
# Two machines since 2026-09-27 (docs/consolidation-plan-2026-09.md):
#
#   Mac mini  collection + inference, every 900s under a system LaunchDaemon
#             (mini/), plus the single Discord bot that writes the labels.
#             8 GB and always on — it never trains.
#   This Air  training on MPS, on demand: `just pull-labels`, `just train`,
#             `just push-checkpoint`. The mini's next tick serves the result.
#
# The site is GitHub Pages. No Nomad, no containers, no Cloudflare.

MOUNTAIN_ROOT := env_var_or_default("MOUNTAIN_ROOT", env_var("HOME") + "/mountain")

# List available recipes
default:
    @just --list

#   just train              # ~/mountain/data/labels.yaml, 5 epochs
#   just train data/labels.yaml 5
#
# Labels default to [paths] labels_dir — whatever `just pull-labels` brought
# down. AIR ONLY: the 8 GB mini must never train.
# Train the ConvNeXt-Tiny LoRA + dual-input classifier (MPS, on demand).
train labels="" epochs="5":
    #!/usr/bin/env bash
    set -euo pipefail
    . air/_config.sh
    mountain_load_config
    labels="{{ labels }}"
    [ -n "$labels" ] || labels="$AIR_LABELS/labels.yaml"
    [ -f "$labels" ] || { echo "no labels at $labels — run 'just pull-labels' first" >&2; exit 1; }
    uv run training batch --labels "$labels" --epochs {{ epochs }}

# Mirror the mini's labels + the captures they reference onto this Mac.
pull-labels *args:
    bash air/pull-labels.sh {{ args }}

# Ship a trained checkpoint to the mini and kickstart the tick to load it.
push-checkpoint *args:
    bash air/push-checkpoint.sh {{ args }}

# One capture + one native inference + publish — exactly what launchd runs.
tick:
    bash mini/tick.sh

# Same tick, but leave GitHub Pages alone (useful on a laptop).
tick-local:
    MOUNTAIN_SKIP_PUBLISH=1 bash mini/tick.sh

# Single capture (webcam frame + METAR) into $MOUNTAIN_ROOT/data.
capture:
    uv run collect capture-once --config mountain.toml --data-root "{{MOUNTAIN_ROOT}}/data"

# One inference into the live dir, without capturing or publishing.
predict:
    uv run python tools/predict_state.py \
        --config mountain.toml \
        --out "{{MOUNTAIN_ROOT}}/live/state.json" \
        --log "{{MOUNTAIN_ROOT}}/live/history.jsonl" \
        --checkpoint-dir "{{MOUNTAIN_ROOT}}/checkpoints"

# Push the mini's current live/ state to GitHub Pages now.
publish:
    gh workflow run publish.yml -R robogeosociety/is-the-mountain-out

# Run this at the mini's console: it sudo's, and a dropped ssh session between
# bootout and bootstrap leaves the tick unloaded with nothing scheduled.
# Install / reinstall the 15-minute LaunchDaemon on the mini.
install-daemon:
    bash mini/install.sh

# Harvest Discord reactions into labels.yaml (needs cf.env — see BOT.md)
bot:
    uv run --group bot bot run --config mountain.toml

# Post a single labelable capture to Discord, then exit (setup check)
bot-post-once:
    uv run --group bot bot post-once --config mountain.toml

# The CI gates, locally.
check:
    uvx ruff@0.16.0 check .
    uvx ruff@0.16.0 format --check .
    uv run pytest -q
    cd web && npm run lint && npm run build

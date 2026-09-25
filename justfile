# is-the-mountain-out tasks.
#
# Everything runs on the Mac mini natively since 2026-09: collection and
# inference on a 15-minute LaunchAgent (mini/), training on demand on MPS,
# the site on GitHub Pages. No Nomad, no containers, no Cloudflare.

MOUNTAIN_ROOT := env_var_or_default("MOUNTAIN_ROOT", "/Volumes/dev/mountain")

# List available recipes
default:
    @just --list

# Train the ConvNeXt-Tiny LoRA + dual-input classifier; best checkpoint → train/checkpoints/.
#   just train data/labels.yaml 5
train labels="data/labels.yaml" epochs="5":
    uv run training batch --labels {{labels}} --epochs {{epochs}}

# One capture + one native inference + publish — exactly what launchd runs.
tick:
    bash mini/tick.sh

# Same tick, but leave GitHub Pages alone (useful on a laptop).
tick-local:
    MOUNTAIN_SKIP_PUBLISH=1 bash mini/tick.sh

# Single capture (webcam frame + METAR) into the dev-disk data root.
capture:
    uv run collect capture-once --config mountain.toml --data-root "{{MOUNTAIN_ROOT}}/data"

# One inference into the live dir, without capturing or publishing.
predict:
    uv run python tools/predict_state.py \
        --config mountain.toml \
        --out "{{MOUNTAIN_ROOT}}/live/state.json" \
        --log "{{MOUNTAIN_ROOT}}/live/history.jsonl" \
        --checkpoint-dir "{{MOUNTAIN_ROOT}}/checkpoints"

# Push the current dev-disk state to GitHub Pages now.
publish:
    gh workflow run publish.yml -R robogeosociety/is-the-mountain-out

# Install / reinstall the 15-minute LaunchAgent (run in a GUI session, not ssh).
install-agent:
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

# is-the-mountain-out tasks. Training runs locally on your most capable machine
# (it prefetches from R2 and runs gradient descent on MPS — see TRAINING.md);
# Nomad is reserved for the always-on collector service, not one-shot training.

# List available recipes
default:
    @just --list

# Train the ConvNeXt-Tiny LoRA + dual-input classifier; best checkpoint → train/checkpoints/.
#   just train data/labels.yaml 5
train labels="data/labels.yaml" epochs="5":
    uv run training batch --labels {{labels}} --epochs {{epochs}}

# Run the Discord reaction-labeling bot: harvests reactions on the Worker's
# visibility notifications into labels.yaml (needs cf.env — see BOT.md)
bot:
    uv run --group bot bot run --config mountain.toml

# Post a single labelable capture to Discord, then exit (setup check)
bot-post-once:
    uv run --group bot bot post-once --config mountain.toml

"""Which camera a checkpoint was trained on — and whether it matches the live one.

A LoRA checkpoint is only meaningful for the view it was fit to. When the
camera changed on 2026-09-24 (UW ATG died; KING 5 Queen Anne replaced it) the
old weights kept loading and kept emitting confident probabilities about a
framing they had never seen. Nothing in the pipeline could tell: the feed was
alive, the tensors were the right shape, the softmax summed to one.

So the contract is made explicit. The camera declares an era
(`[webcam] era` in mountain.toml) and every checkpoint records the era it was
trained under (`era.json` beside the weights, written at save time). When they
disagree, `tools/predict_state.py` publishes `status: "unvalidated"` and no
prediction at all — the site says CHECKING… until a checkpoint for this camera
exists.

Old checkpoints predate the marker, so a missing `era.json` falls back to
`[training] checkpoint_era` in the config. That is how the currently-live
UW-era weights are correctly recognised as stale without anyone touching the
mini.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

#: Filename written beside the weights.
ERA_FILENAME = "era.json"


def read_checkpoint_era(
    checkpoint_dir: str | Path, fallback: Optional[str] = None
) -> Optional[str]:
    """Era recorded beside the weights, else *fallback*, else None.

    A malformed or unreadable marker is treated as absent: the fallback is a
    deliberate statement about untagged checkpoints, and a corrupt file should
    not silently promote one to "matching".
    """
    path = Path(checkpoint_dir) / ERA_FILENAME
    try:
        if path.exists():
            era = json.loads(path.read_text()).get("era")
            if isinstance(era, str) and era.strip():
                return era.strip()
    except OSError, ValueError:
        pass
    return fallback


def write_checkpoint_era(checkpoint_dir: str | Path, era: str, **extra) -> Path:
    """Stamp *checkpoint_dir* with the era it was trained under.

    Called by the trainer every time it saves, so a checkpoint is never
    ambiguous about which camera it belongs to.
    """
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    marker = path / ERA_FILENAME
    # default=str: an unserializable extra must never be the thing that blows
    # up a checkpoint save. The era itself is validated by the caller.
    marker.write_text(
        json.dumps({"era": era, **extra}, indent=2, sort_keys=True, default=str) + "\n"
    )
    return marker


def era_matches(camera_era: Optional[str], checkpoint_era: Optional[str]) -> bool:
    """Is this checkpoint valid for this camera?

    Unknown on either side is NOT a match. "We do not know" and "yes" are
    different answers, and only one of them should put a number on the site.
    An unset `[webcam] era` disables the gate entirely — the caller checks that
    separately, so this stays a pure comparison.
    """
    if not camera_era or not checkpoint_era:
        return False
    return camera_era.strip() == checkpoint_era.strip()

# `mini/` — the whole production runtime

Since 2026-09 the mountain runs on the Mac mini and GitHub Pages. There is no
Cloudflare Worker, no container, no Nomad job and no cron in the cloud.

```
launchd (every 900s)
  └─ ~/.local/libexec/mountain-tick.sh        (copy of tick.sh)
       ├─ bounded probe of /Volumes/dev       ← bails if the disk is wedged
       ├─ uv run collect capture-once         → /Volumes/dev/mountain/data/…
       ├─ uv run python tools/predict_state.py (native MPS, local checkpoint)
       │    → live/state.json, live/history.jsonl, live/announce.jsonl
       └─ gh workflow run publish.yml         → GitHub Pages
```

| File | Role |
| --- | --- |
| `tick.sh` | one tick: probe, capture, infer, publish |
| `com.robogeosociety.mountain-tick.plist` | LaunchAgent, `StartInterval 900`, `RunAtLoad` |
| `install.sh` | render + copy the plist, copy the script to `~/.local/libexec`, bootstrap, verify |
| `r2-pull.sh` | one-shot: copy both R2 buckets onto the dev disk |

## Layout on the dev disk

```
/Volumes/dev/mountain/
├── data/          captures (YYYYMMDD/HHMMSS_us_UTC/{images,metar}) + labels.yaml
├── checkpoints/   adapter_config.json, adapter_model.safetensors, classifier.pt
└── live/          state.json · history.jsonl · announce.jsonl · notify-state.json
```

`live/state.json` and `live/history.jsonl` are what `publish.yml` copies into
the Pages artifact. `live/announce.jsonl` is the Discord queue — append-only,
one JSON object per line, `posted: false`; the single Discord bot drains it and
posts the alert or label request. The tick never calls Discord itself.

## First install (after the dev disk is back)

```sh
cd /Volumes/dev/is-the-mountain-out
bash mini/r2-pull.sh --dry-run   # check what would land
bash mini/r2-pull.sh             # ~once, then R2 is out of the path
bash mini/install.sh             # IN A GUI SESSION, not over ssh
launchctl kickstart -p gui/$(id -u)/com.robogeosociety.mountain-tick
tail -f ~/Library/Logs/mountain-tick.out.log
```

`install.sh` must run from a logged-in GUI session: `launchctl bootstrap
gui/$UID` fails over ssh while the paired `bootout` succeeds, which is how
agents on this mini have ended up silently unloaded for days. The script
verifies with `launchctl print` and exits non-zero if the job did not load.

The agent runs the **copy** in `~/.local/libexec`, not the repo file — that is
what keeps a wedged `/Volumes/dev` from hanging the exec itself, before
`tick.sh`'s own probe can bail. Re-run `install.sh` after editing `tick.sh`.

## Knobs

All optional environment variables, read by `tick.sh`:

| Variable | Default |
| --- | --- |
| `MOUNTAIN_DEV_ROOT` | `/Volumes/dev` |
| `MOUNTAIN_ROOT` | `$MOUNTAIN_DEV_ROOT/mountain` |
| `MOUNTAIN_REPO` | `$MOUNTAIN_DEV_ROOT/is-the-mountain-out` |
| `MOUNTAIN_LIVE` / `MOUNTAIN_DATA` / `MOUNTAIN_CHECKPOINTS` | under `$MOUNTAIN_ROOT` |
| `MOUNTAIN_DEVICE` | `mps` (falls back to CPU automatically) |
| `MOUNTAIN_PROBE_TIMEOUT` | `5` seconds |
| `MOUNTAIN_SKIP_PUBLISH` | unset; `1` runs a tick without dispatching Pages |

## Troubleshooting

```sh
launchctl print gui/$(id -u)/com.robogeosociety.mountain-tick   # loaded? last exit status?
bash mini/tick.sh                                               # run it by hand, same code path
MOUNTAIN_SKIP_PUBLISH=1 bash mini/tick.sh                       # …without touching Actions
tail -n 50 ~/Library/Logs/mountain-tick.err.log
```

- **"disk is wedged, skipping this tick"** — the bounded probe did its job.
  Detach, reboot, reattach the dev disk; the next tick recovers on its own.
- **Inference fails but the site looks fine** — `state.json` is only rewritten
  on success, and `history.jsonl` carries the traceback. The SPA marks itself
  stale after an hour.
- **Nothing publishes** — `gh auth status`. Dispatch is best-effort; the
  `*/15` schedule in `publish.yml` is the backstop.

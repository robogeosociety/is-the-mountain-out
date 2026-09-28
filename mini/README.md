# `mini/` — the whole serving runtime

Since 2026-09 the mountain runs on the Mac mini and GitHub Pages. There is no
Cloudflare Worker, no container, no Nomad job and no cron in the cloud. The
mini **collects and serves**; it does not train — training is Air-only
(`TRAINING.md`, `air/`).

```
launchd  system/com.robogeosociety.mountain-tick   (LaunchDaemon, every 900s)
  └─ /usr/local/libexec/mountain-tick.sh           (copy of tick.sh, root:wheel)
       │  runs as tommydoerr:staff, no user session required
       ├─ bounded probe of /Volumes/dev            ← only if a path still lives there
       ├─ uv run collect capture-once              → ~/mountain/data/…
       ├─ uv run python tools/predict_state.py     (native MPS, local checkpoint)
       │    → live/state.json, live/history.jsonl, live/announce.jsonl
       └─ gh workflow run publish.yml              → GitHub Pages
```

| File | Role |
| --- | --- |
| `tick.sh` | one tick: probe, capture, infer, publish |
| `launchd/com.robogeosociety.mountain-tick.plist` | **LaunchDaemon**, `StartInterval 900`, `RunAtLoad`, `UserName tommydoerr` |
| `install.sh` | copy the script + plist as root, `bootstrap system`, verify |
| `r2-pull.sh` | one-shot: copy both R2 buckets onto local disk |

## Why a daemon and not an agent (2026-09-27)

The mini is headless with FileVault on. The only remote reboot is
`sudo fdesetup authrestart`, which unlocks the disk once and then boots to the
**login window** — LaunchDaemons run, LaunchAgents do not, because there is no
user session for them to live in. As an agent the tick stayed dark after every
unattended reboot and nothing said so.

So the tick is a system daemon that drops to `tommydoerr:staff`. Three
consequences worth knowing:

- **It has no inherited environment.** Everything a login shell would have
  supplied — `HOME`, `PATH`, the `MOUNTAIN_*` roots — is written out in
  `EnvironmentVariables` in the plist. A daemon's `PATH` is shorter than an
  agent's and `HOME` defaults to `/var/root`.
- **It gets no per-user `$TMPDIR`,** so the plist pins `MOUNTAIN_LOCK_DIR` to
  `/tmp/mountain-tick.lock`. Without that, a by-hand `bash mini/tick.sh` would
  take a different lock from the daemon and two ticks could run at once.
- **`launchctl` verbs need root:** `sudo launchctl kickstart -p
  system/com.robogeosociety.mountain-tick`, not the `gui/$(id -u)` form.
  `air/push-checkpoint.sh` tries `sudo -n` for exactly this and degrades to
  "the next tick will load it" when the sudoers line is absent.

The `fdesetup` half needs one sudoers line on the mini, which is an operator
task, not something this repo installs:

```
tommydoerr ALL=(root) NOPASSWD: /usr/bin/fdesetup authrestart
```

## Where the data lives

The tick's root is on the **boot volume**:

```
/Users/tommydoerr/mountain/
├── data/          captures (YYYYMMDD/HHMMSS_us_UTC/{images,metar}) + labels.yaml + labels/
├── checkpoints/   adapter_config.json, adapter_model.safetensors, classifier.pt, era.json
└── live/          state.json · history.jsonl · announce.jsonl · notify-state.json
```

It was `/Volumes/dev/mountain/` until the 2026-09-27 disk plan. That external
disk has wedged three times (2026-07, 2026-08, 2026-09), and each time it took
the site with it; the mountain's working set is a few GB and belongs on the
disk the machine boots from. **Move the data and the config together** —
`[bot] announce_queue` in `mountain.toml` names the same `live/` directory, and
a tick writing one path while the bot drains another is a silent failure.

`live/state.json` and `live/history.jsonl` are what `publish.yml` copies into
the Pages artifact. `live/announce.jsonl` is the Discord queue — append-only,
one JSON object per line, `posted: false`; the single Discord bot drains it and
posts the alert or label request. The tick never calls Discord itself.

The bounded `/Volumes/dev` probe stays in `tick.sh` and now runs only when one
of the configured paths is still under the dev root (the repo checkout usually
is). Once everything is on the boot volume it is a no-op — but the disk is
still attached, and one environment variable can put a path back under it.

## Install

```sh
cd /Volumes/dev/is-the-mountain-out
bash mini/r2-pull.sh --dry-run   # only if the data is still in R2
bash mini/r2-pull.sh
bash mini/install.sh             # AT THE CONSOLE, as tommydoerr, not under sudo
sudo launchctl kickstart -p system/com.robogeosociety.mountain-tick
tail -f ~/Library/Logs/mountain-tick.out.log
```

`install.sh` calls `sudo` itself for the three steps that need root (the
script into `/usr/local/libexec`, the plist into `/Library/LaunchDaemons`, the
`bootstrap`). Run it **at the console**, not over ssh: `bootstrap system` does
work over ssh — unlike the old `gui/$UID` form — but the script bootouts first,
so a session that drops in between leaves the tick unloaded with nothing
scheduled, and a first-run TCC prompt can only be answered by a logged-in
session. The script verifies with `launchctl print` and exits non-zero if the
job did not load.

Then prove the point of the whole exercise: `sudo fdesetup authrestart`, wait,
and check that `mountain-tick.out.log` grows without anyone logging in.

The daemon runs the **copy** in `/usr/local/libexec`, not the repo file — that
is what keeps a wedged `/Volumes/dev` from hanging the exec itself, before
`tick.sh`'s own probe can bail. Re-run `install.sh` after editing `tick.sh`.

> [!NOTE]
> `gh` must be authenticated for the *daemon's* user with a token it can read
> without an unlocked login keychain. `gh auth login` writes
> `~/.config/gh/hosts.yml` in plain text by default, which is readable at the
> login window; a keychain-backed token is not. A failed dispatch only delays
> the site — `publish.yml` keeps its `*/15` backstop — but it should not
> surprise anyone.

## Knobs

All optional environment variables, read by `tick.sh` and set explicitly in
the plist:

| Variable | Default |
| --- | --- |
| `MOUNTAIN_ROOT` | `$HOME/mountain` |
| `MOUNTAIN_DEV_ROOT` | `/Volumes/dev` (probed only if something still lives there) |
| `MOUNTAIN_REPO` | `$MOUNTAIN_DEV_ROOT/is-the-mountain-out` |
| `MOUNTAIN_LIVE` / `MOUNTAIN_DATA` / `MOUNTAIN_CHECKPOINTS` | under `$MOUNTAIN_ROOT` |
| `MOUNTAIN_DEVICE` | `mps` (falls back to CPU automatically) |
| `MOUNTAIN_PROBE_TIMEOUT` | `5` seconds |
| `MOUNTAIN_LOCK_DIR` | `${TMPDIR:-/tmp}/mountain-tick.lock`; the plist pins `/tmp/…` |
| `MOUNTAIN_SKIP_PUBLISH` | unset; `1` runs a tick without dispatching Pages |

## Troubleshooting

```sh
sudo launchctl print system/com.robogeosociety.mountain-tick   # loaded? last exit status?
bash mini/tick.sh                                              # run it by hand, same code path
MOUNTAIN_SKIP_PUBLISH=1 bash mini/tick.sh                      # …without touching Actions
tail -n 50 ~/Library/Logs/mountain-tick.err.log
```

- **Nothing ran after a reboot** — `sudo launchctl print system/…`. If the
  label is unknown, the plist is not in `/Library/LaunchDaemons` or a
  `bootout` was never followed by a `bootstrap`. Re-run `install.sh` at the
  console.
- **"disk is wedged, skipping this tick"** — the bounded probe did its job.
  Detach, reboot, reattach the dev disk; the next tick recovers on its own.
  Once the repo also lives on the boot volume this cannot fire at all.
- **Inference fails but the site looks fine** — `state.json` is only rewritten
  on success, and `history.jsonl` carries the traceback. The SPA marks itself
  stale after an hour.
- **Nothing publishes** — `gh auth status`. Dispatch is best-effort; the
  `*/15` schedule in `publish.yml` is the backstop.
- **The site says CHECKING… and `state.json` has `status: "unvalidated"`** —
  working as intended: no checkpoint has been trained on the current camera, so
  no prediction is computed (`train/checkpoint_era.py`). Frames still go to
  Discord for labeling. It clears itself when the Air pushes the first Queen
  Anne checkpoint (`air/push-checkpoint.sh`) — no config change, no deploy.

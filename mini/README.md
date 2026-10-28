# `mini/` — the whole serving runtime

Since 2026-09 the mountain runs on the Mac mini and GitHub Pages. There is no
Cloudflare Worker, no container, no Nomad job and no cron in the cloud. The
mini **collects and serves**; it does not train — training is Air-only
(`TRAINING.md`, `air/`).

```
launchd  system/com.robogeosociety.mountain-tick   (LaunchDaemon, every 900s)
  └─ /usr/local/libexec/mountain-tick.sh           (copy of tick.sh, root:wheel)
       │  runs as tommydoerr:staff, no user session required
       ├─ uv run collect capture-once              → ~/mountain/data/…
       ├─ uv run python tools/predict_state.py     (native MPS, local checkpoint)
       │    → live/state.json, live/history.jsonl, live/announce.jsonl
       └─ gh workflow run publish.yml              → GitHub Pages
```

| File | Role |
| --- | --- |
| `tick.sh` | one tick: capture, infer, publish |
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

Two sudoers lines on the mini make the unattended half work. They are operator
tasks, not something this repo installs:

```
tommydoerr ALL=(root) NOPASSWD: /usr/bin/fdesetup authrestart
tommydoerr ALL=(root) NOPASSWD: /bin/launchctl kickstart -k system/com.robogeosociety.mountain-tick
```

The first is the remote reboot. The second lets `air/push-checkpoint.sh` load a
freshly trained checkpoint immediately instead of waiting out a quarter hour
(`launchctl` is `/bin/launchctl` on macOS). Without either, the fallbacks are a
hand on the keyboard and a 15-minute wait respectively — neither is broken,
both are slower.

## Where everything lives

**The boot volume, all of it.** The 2026-09-27 disk plan puts everything
critical on the internal disk and leaves only overflow on `/Volumes/dev` —
the external disk that has wedged three times (2026-07, 2026-08, 2026-09) and
took the site with it each time.

```
/Users/tommydoerr/
├── mountain/
│   ├── data/          captures (YYYYMMDD/HHMMSS_us_UTC/{images,metar}) + labels.yaml + labels/
│   ├── checkpoints/   adapter_config.json, adapter_model.safetensors, classifier.pt, era.json
│   └── live/          state.json · history.jsonl · announce.jsonl · notify-state.json
└── src/
    └── is-the-mountain-out/   the checkout the tick runs `uv run` in (MOUNTAIN_REPO)
```

**Move the data and the config together** — `[bot] announce_queue` in
`mountain.toml` names the same `live/` directory, and a tick writing one path
while the bot drains another is a silent failure. Same for `publish.yml`'s
`MOUNTAIN_LIVE`.

Because nothing the tick touches is under `/Volumes` any more, **`tick.sh` no
longer probes the dev disk at all** — the bounded child-process probe that used
to open every tick is deleted, along with `MOUNTAIN_DEV_ROOT` and
`MOUNTAIN_PROBE_TIMEOUT`. A guard for a path that cannot occur is a guard
nobody maintains. (`publish.yml` keeps its equivalent, because `MOUNTAIN_LIVE`
is a repo variable someone could point anywhere.)

`live/state.json` and `live/history.jsonl` are what `publish.yml` copies into
the Pages artifact. `live/announce.jsonl` is the Discord queue — append-only,
one JSON object per line, `posted: false`; the single Discord bot drains it and
posts the alert or label request. The tick never calls Discord itself.

## Install

The checkout has to be on the boot volume first — the daemon's `MOUNTAIN_REPO`
is `/Users/tommydoerr/src/is-the-mountain-out`, and `uv run` happens there:

```sh
mkdir -p ~/src && git clone https://github.com/robogeosociety/is-the-mountain-out ~/src/is-the-mountain-out
# …or move an existing one: mv /Volumes/dev/is-the-mountain-out ~/src/

cd ~/src/is-the-mountain-out
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

The daemon runs the **copy** in `/usr/local/libexec`, not the repo file, so
the 15-minute schedule does not depend on a working tree somebody is mid-rebase
on. Re-run `install.sh` after editing `tick.sh`.

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
| `MOUNTAIN_REPO` | `$HOME/src/is-the-mountain-out` |
| `MOUNTAIN_LIVE` / `MOUNTAIN_DATA` / `MOUNTAIN_CHECKPOINTS` | under `$MOUNTAIN_ROOT` |
| `MOUNTAIN_DEVICE` | `mps` (falls back to CPU automatically) |
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
- **"repo not found at …"** — the checkout is not at `MOUNTAIN_REPO`. Clone or
  move it to `~/src/is-the-mountain-out`; the daemon does not look anywhere
  else. (The old "disk is wedged" bail is gone with the dev-disk dependency.)
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

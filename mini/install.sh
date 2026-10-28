#!/bin/bash
# Install the mountain tick LaunchDaemon on the Mac mini.
#
#   bash mini/install.sh            # install / reinstall
#   bash mini/install.sh --uninstall
#
# RUN THIS AT THE MINI'S CONSOLE (keyboard or Screen Sharing), not over ssh,
# and as tommydoerr — the script calls `sudo` itself, so do NOT prefix it with
# sudo.
#
# `bootstrap system` does work over ssh, unlike the `gui/$UID` bootstrap the
# old LaunchAgent needed. Run it at the console anyway, for two reasons:
#
#   1. This script bootouts before it bootstraps. A dropped ssh session
#      between the two leaves the tick UNLOADED — nothing scheduled, no error
#      anywhere — which is exactly how agents on this mini have gone dark for
#      days at a time.
#   2. The first run of a new daemon can raise a TCC prompt (Files and
#      Folders). Only a logged-in session can answer one; over ssh it is
#      denied silently.
#
# Why a daemon at all: the mini is headless with FileVault on, and the remote
# reboot is `sudo fdesetup authrestart`, which boots to the login window with
# no user session. LaunchAgents do not run there. See mini/README.md.
set -euo pipefail

LABEL="com.robogeosociety.mountain-tick"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLIST_SRC="$SRC_DIR/launchd/$LABEL.plist"
DAEMON_PLIST="/Library/LaunchDaemons/$LABEL.plist"
LIBEXEC_DIR="/usr/local/libexec"
INSTALLED_SCRIPT="$LIBEXEC_DIR/mountain-tick.sh"
DOMAIN="system"

uninstall() {
    sudo launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    sudo rm -f "$DAEMON_PLIST" "$INSTALLED_SCRIPT"
    echo "Removed $LABEL (plist, script). Logs in ~/Library/Logs are kept."
}

if [ "${1:-}" = "--uninstall" ]; then
    uninstall
    exit 0
fi

[ -f "$PLIST_SRC" ] || { echo "ERROR: $PLIST_SRC not found." >&2; exit 1; }

# The plist names the account the daemon drops to. Read it back rather than
# assuming: installing as the wrong user would give the tick a home directory
# it cannot write and logs nobody reads.
RUN_USER="$(/usr/libexec/PlistBuddy -c 'Print :UserName' "$PLIST_SRC" 2>/dev/null || true)"
[ -n "$RUN_USER" ] || { echo "ERROR: $PLIST_SRC has no UserName." >&2; exit 1; }
RUN_HOME="$(/usr/libexec/PlistBuddy -c 'Print :EnvironmentVariables:HOME' "$PLIST_SRC")"

if [ "$(id -un)" != "$RUN_USER" ]; then
    echo "ERROR: run this as $RUN_USER (the plist's UserName), not $(id -un)." >&2
    echo "       Do not run the script under sudo either — it sudo's the three" >&2
    echo "       steps that need root and keeps the rest as the tick's own user." >&2
    exit 1
fi
[ -d "$RUN_HOME" ] || { echo "ERROR: $RUN_HOME does not exist." >&2; exit 1; }

plutil -lint "$PLIST_SRC" >/dev/null

# The tick script is copied out of the checkout, root-owned, next to the
# plist. launchd exec's it every 15 minutes, and the schedule should not depend
# on a working tree somebody is mid-rebase on. See the plist header.
sudo mkdir -p "$LIBEXEC_DIR"
sudo install -m 0755 -o root -g wheel "$SRC_DIR/tick.sh" "$INSTALLED_SCRIPT"

# Logs are written as $RUN_USER, so the directory must exist and be theirs.
mkdir -p "$RUN_HOME/Library/Logs"

sudo install -m 0644 -o root -g wheel "$PLIST_SRC" "$DAEMON_PLIST"

# Reload. bootout first (ignore "not loaded"), then bootstrap — and check.
sudo launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
if ! sudo launchctl bootstrap "$DOMAIN" "$DAEMON_PLIST"; then
    echo "ERROR: launchctl bootstrap $DOMAIN failed." >&2
    echo "       The daemon is now UNLOADED, not merely un-updated. Fix the" >&2
    echo "       plist and re-run; nothing is scheduled until this succeeds." >&2
    exit 1
fi

if ! sudo launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
    echo "ERROR: $LABEL is not loaded after bootstrap. Nothing is scheduled." >&2
    exit 1
fi

cat <<EOF
Installed $LABEL
  plist   $DAEMON_PLIST (root:wheel 0644)
  script  $INSTALLED_SCRIPT (copied from $SRC_DIR/tick.sh)
  runs as $RUN_USER:staff, home $RUN_HOME
  logs    $RUN_HOME/Library/Logs/mountain-tick.{out,err}.log
  every   900s, and once now (RunAtLoad)

Next:
  sudo launchctl kickstart -p $DOMAIN/$LABEL   # run a tick now
  tail -f $RUN_HOME/Library/Logs/mountain-tick.out.log

It survives a reboot with no login — that is the point. Verify once with
\`sudo fdesetup authrestart\` and check the log timestamps afterwards.

Reinstall after editing tick.sh — the daemon runs the COPY, not the repo file.
EOF

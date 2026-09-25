#!/bin/bash
# Install the mountain tick LaunchAgent on the Mac mini.
#
#   bash mini/install.sh            # install / reinstall
#   bash mini/install.sh --uninstall
#
# RUN THIS IN A LOGGED-IN GUI SESSION (Screen Sharing or at the keyboard), not
# over ssh. `launchctl bootstrap gui/$UID` needs the user's GUI domain; over ssh
# it fails with "Bootstrap failed: 5: Input/output error" — and the bootout it
# is paired with DOES succeed, which is how agents have ended up silently
# unloaded for days. This script therefore verifies with `launchctl print`
# afterwards and exits non-zero if the job is not actually loaded.
set -euo pipefail

LABEL="com.robogeosociety.mountain-tick"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$HOME/Library/LaunchAgents"
AGENT_PLIST="$AGENT_DIR/$LABEL.plist"
LIBEXEC_DIR="$HOME/.local/libexec"
INSTALLED_SCRIPT="$LIBEXEC_DIR/mountain-tick.sh"
DOMAIN="gui/$(id -u)"

uninstall() {
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    rm -f "$AGENT_PLIST" "$INSTALLED_SCRIPT"
    echo "Removed $LABEL (plist, script). Logs in ~/Library/Logs are kept."
}

if [ "${1:-}" = "--uninstall" ]; then
    uninstall
    exit 0
fi

# The tick script is copied to the BOOT disk. launchd exec's it every 15
# minutes; run from /Volumes/dev it would hang on a wedged disk before the
# script's own bounded probe could bail. See the plist header.
mkdir -p "$LIBEXEC_DIR" "$AGENT_DIR" "$HOME/Library/Logs"
install -m 0755 "$SRC_DIR/tick.sh" "$INSTALLED_SCRIPT"

# A plist cannot expand $HOME, so render the placeholder at install time.
sed "s|__HOME__|$HOME|g" "$SRC_DIR/$LABEL.plist" > "$AGENT_PLIST"
chmod 0644 "$AGENT_PLIST"
plutil -lint "$AGENT_PLIST" >/dev/null

# Reload. bootout first (ignore "not loaded"), then bootstrap — and check.
launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
if ! launchctl bootstrap "$DOMAIN" "$AGENT_PLIST"; then
    echo "ERROR: launchctl bootstrap $DOMAIN failed." >&2
    echo "       If you are on ssh, run this again from a GUI session; the agent" >&2
    echo "       is now UNLOADED, not merely un-updated." >&2
    exit 1
fi

if ! launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
    echo "ERROR: $LABEL is not loaded after bootstrap. Nothing is scheduled." >&2
    exit 1
fi

cat <<EOF
Installed $LABEL
  plist   $AGENT_PLIST
  script  $INSTALLED_SCRIPT (copied from $SRC_DIR/tick.sh)
  logs    ~/Library/Logs/mountain-tick.{out,err}.log
  every   900s, and once now (RunAtLoad)

Next:
  launchctl kickstart -p $DOMAIN/$LABEL   # run a tick now
  tail -f ~/Library/Logs/mountain-tick.out.log

Reinstall after editing tick.sh — the agent runs the COPY, not the repo file.
EOF

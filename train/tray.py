"""
TrainingTray — rumps menu bar app for monitoring Nomad training jobs.

Reads data/training_state.json via a rumps.Timer.
"""

import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any

import rumps

logger = logging.getLogger(__name__)

REFRESH_INTERVAL = 10  # seconds between state file reads


SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class TrainingTray(rumps.App):
    def __init__(self, data_root: str = "data"):
        super().__init__("Mountain Training", title="🗻", quit_button=None)
        self.data_root = Path(data_root).absolute()
        self.state_file = self.data_root / "training_state.json"
        self._last_state: Optional[Dict[str, Any]] = None
        self._spinner_idx = 0

        # --- Static menu skeleton ---
        self.progress_bar_item = rumps.MenuItem("—")
        self.status_item = rumps.MenuItem("Status: —")
        self.epoch_item = rumps.MenuItem("Epoch: —")
        self.batch_item = rumps.MenuItem("Batch: —")
        self.loss_item = rumps.MenuItem("Loss: —")

        self.menu = [
            self.progress_bar_item,
            rumps.separator,
            self.status_item,
            rumps.separator,
            self.epoch_item,
            self.batch_item,
            self.loss_item,
            rumps.separator,
            rumps.MenuItem("Quit Monitor", callback=self._on_quit),
        ]

        self._timer = rumps.Timer(self._refresh, REFRESH_INTERVAL)

    def _read_state(self) -> Optional[Dict[str, Any]]:
        if not self.state_file.exists():
            return None
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read state file: {e}")
            return None

    def _refresh(self, _=None) -> None:
        state = self._read_state()
        if state is None:
            self.status_item.title = "Status: No training state found"
            self.title = "🗻"
            return

        # Check if state changed (or if it's running, we just update to spin the icon if needed)
        if state == self._last_state and state.get("status") != "running":
            return

        self._last_state = state
        self._render(state)

    def _render(self, state: Dict[str, Any]) -> None:
        status = state.get("status", "unknown")

        if status == "running":
            # Simple spinner logic by cycling through braille frames
            spinner = SPINNER[self._spinner_idx % len(SPINNER)]
            self._spinner_idx += 1
            self.title = f"🗻{spinner}"
            self.status_item.title = "Status: 🟢 Running"
        elif status == "complete":
            self.title = "🗻"
            self.status_item.title = "Status: ⚪️ Complete"
        else:
            self.title = "🗻"
            self.status_item.title = f"Status: {status}"

        # Progress bar (based on batches)
        batches = state.get("batches_complete", 0)
        total_batches = state.get("total_batches", 1)  # Avoid div zero
        pct = int(100 * batches / total_batches) if total_batches > 0 else 0

        bar_len = 20
        filled_len = int(round(bar_len * pct / 100))
        bar = "█" * filled_len + "░" * (bar_len - filled_len)
        self.progress_bar_item.title = f"[{bar}] {pct}%"

        # Epochs
        epoch = state.get("epoch", 0)
        total_epochs = state.get("total_epochs", 0)
        self.epoch_item.title = f"Epoch: {epoch}/{total_epochs}"

        # Batches
        self.batch_item.title = f"Batch: {batches}/{total_batches}"

        # Loss
        loss = state.get("current_loss", 0.0)
        self.loss_item.title = f"Loss: {loss:.4f}"

    def _on_quit(self, _):
        logger.info("Quitting tray...")
        rumps.quit_application()

    def run(self):
        self._refresh()
        self._timer.start()
        super().run()


def cli():
    logging.basicConfig(level=logging.INFO)
    TrainingTray().run()


if __name__ == "__main__":
    cli()

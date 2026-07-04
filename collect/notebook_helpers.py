import yaml
import json
from pathlib import Path
from IPython.display import display
import ipywidgets as widgets


def load_labels(data_root):
    """Loads the training index from labels.yaml."""
    labels_path = Path(data_root) / "labels.yaml"
    if labels_path.exists():
        with open(labels_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_labels(data_root, labels):
    """Saves the training index to labels.yaml."""
    labels_path = Path(data_root) / "labels.yaml"
    with open(labels_path, "w") as f:
        yaml.safe_dump(labels, f)


class CaptureBrowser:
    def __init__(self, log_path="data/collection.log", data_root="data", batch_size=20):
        self.log_path = Path(log_path)
        self.data_root = Path(data_root)
        self.batch_size = batch_size

        # UI Components
        self.main_container = widgets.VBox()
        self.grid_container = widgets.GridBox(
            layout=widgets.Layout(
                grid_template_columns="repeat(2, 512px)",
                grid_gap="20px",
                justify_content="center",
            )
        )
        self.status_label = widgets.HTML()
        self.control_bar = widgets.HBox(
            layout=widgets.Layout(justify_content="center", margin="20px 0")
        )

        self.all_captures = []
        self.current_page = 0

    def _load_captures(self):
        """Loads all captures from the log file, newest first."""
        captures = []
        if not self.log_path.exists():
            return []

        with open(self.log_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if (
                        entry.get("event") == "CAPTURE"
                        and entry.get("status") == "SUCCESS"
                    ):
                        meta = entry["metadata"]
                        captures.append(
                            {
                                "timestamp": entry["timestamp"],
                                "image": meta["image_path"],
                                "metar": meta.get("metar_path"),
                            }
                        )
                except Exception:
                    continue
        return sorted(captures, key=lambda x: x["timestamp"], reverse=True)

    def refresh_ui(self):
        self.all_captures = self._load_captures()

        start = self.current_page * self.batch_size
        end = start + self.batch_size
        page_items = self.all_captures[start:end]

        self.status_label.value = f"<div style='text-align:center;'><b>Total Captures:</b> {len(self.all_captures)} | <b>Showing:</b> {start + 1}-{min(end, len(self.all_captures))}</div>"

        if not page_items:
            self.grid_container.children = [
                widgets.HTML(
                    "<h3 style='text-align:center; width:100%;'>No captures found in log.</h3>"
                )
            ]
            return

        grid_items = []
        for item in page_items:
            abs_img = Path(item["image"])
            if not abs_img.exists():
                abs_img = self.data_root.parent / item["image"]

            try:
                with open(abs_img, "rb") as f:
                    img_widget = widgets.Image(value=f.read(), format="jpg", width=512)

                metar_text = "N/A"
                if item["metar"]:
                    metar_p = Path(item["metar"])
                    if not metar_p.exists():
                        metar_p = self.data_root.parent / item["metar"]
                    if metar_p.exists():
                        with open(metar_p, "r") as f:
                            metar_text = f.read().strip()

                info = widgets.HTML(
                    f"<div style='font-size:11px; color:#666;'>{item['timestamp']}<br><code>{metar_text}</code></div>"
                )
                grid_items.append(widgets.VBox([img_widget, info]))
            except Exception:
                continue

        self.grid_container.children = grid_items

        # Pagination buttons
        btn_prev = widgets.Button(
            description="⬅️ Newer", disabled=(self.current_page == 0)
        )
        btn_next = widgets.Button(
            description="Older ➡️", disabled=(end >= len(self.all_captures))
        )

        def go_prev(_):
            self.current_page -= 1
            self.refresh_ui()

        def go_next(_):
            self.current_page += 1
            self.refresh_ui()

        btn_prev.on_click(go_prev)
        btn_next.on_click(go_next)
        self.control_bar.children = [btn_prev, btn_next]

    def start(self):
        display(self.main_container)
        self.main_container.children = [
            widgets.HTML("<h1 style='text-align:center;'>🏔️ Capture Monitor</h1>"),
            self.status_label,
            widgets.HTML("<hr>"),
            self.grid_container,
            widgets.HTML("<hr>"),
            self.control_bar,
        ]
        self.refresh_ui()

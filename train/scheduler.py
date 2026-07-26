import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch
import typer
from torch import optim

from train.config_loader import ConfigLoader
from train.model import ConvNextLoRAModel
from train.utils import WeatherFetcher, WebcamStream

app = typer.Typer()


class Trainer:
    def __init__(self, config_path: str = "mountain.toml", fresh: bool = False):
        self.config_loader = ConfigLoader(config_path)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        # Initialize components
        self.model_wrapper = ConvNextLoRAModel(
            num_classes=3,
            rank=self.config_loader.lora_settings["rank"],
            alpha=self.config_loader.lora_settings["alpha"],
            target_modules=self.config_loader.lora_settings["target_modules"],
            device=self.device,
        )

        # Attempt to load checkpoint (skip if fresh training requested)
        if not fresh:
            self.model_wrapper.load_checkpoint(self.config_loader.checkpoint_dir)

        # Lower learning rate for fine-tuning stability
        self.optimizer = optim.Adam(
            self.model_wrapper.model_dict.parameters(), lr=0.0001
        )
        self.weather_fetcher = WeatherFetcher(self.config_loader.metar_station)

    def run_single_cycle(self, label: int = 1):
        print(f"[{datetime.now()}] Starting single training cycle...")
        weather_vector = self.weather_fetcher.get_weather_vector()

        source = self.config_loader.webcam_url
        stream = WebcamStream(source, device=self.device)
        try:
            tensor = stream.capture_to_tensor()
            if tensor is not None:
                # Still use batch format even for single image
                image_batch = tensor
                weather_batch = weather_vector.unsqueeze(0)
                label_batch = torch.tensor([label]).to(self.device)

                loss = self.model_wrapper.train_step(
                    image_batch, weather_batch, label_batch, self.optimizer
                )
                print(f"[{datetime.now()}] Cycle Complete: Loss = {loss:.4f}")
                self.model_wrapper.save_checkpoint(self.config_loader.checkpoint_dir)
            else:
                print(f"  Source {source}: Capture failed.")
        finally:
            stream.release()

    def live_training_loop(self, label: int = 1):
        print(f"[{datetime.now()}] Starting continuous live training loop...")
        image_list = []
        weather_list = []
        label_list = []

        source = self.config_loader.webcam_url
        try:
            while True:
                weather_vector = self.weather_fetcher.get_weather_vector()
                stream = WebcamStream(source, device=self.device)
                try:
                    tensor = stream.capture_to_tensor()
                    if tensor is not None:
                        image_list.append(tensor.squeeze(0))
                        weather_list.append(weather_vector)
                        label_list.append(torch.tensor(label))
                        print(f"  Captured from {source}")
                    else:
                        print(f"  Source {source}: Capture failed.")
                finally:
                    stream.release()

                if image_list:
                    current_accum = len(image_list)
                    print(
                        f"  Accumulation step {current_accum}/{self.config_loader.gradient_accumulation_steps}"
                    )

                    if current_accum >= self.config_loader.gradient_accumulation_steps:
                        image_batch = torch.stack(image_list)
                        weather_batch = torch.stack(weather_list)
                        label_batch = torch.stack(label_list)
                        loss = self.model_wrapper.train_step(
                            image_batch, weather_batch, label_batch, self.optimizer
                        )
                        print(
                            f"[{datetime.now()}] Batch Training Complete: Loss = {loss:.4f}"
                        )
                        self.model_wrapper.save_checkpoint(
                            self.config_loader.checkpoint_dir
                        )
                        image_list, weather_list, label_list = [], [], []

                time.sleep(self.config_loader.capture_interval_seconds)
        # Keep the parens 3.13-compatible; ruff's py314 style would strip them.
        except (KeyboardInterrupt, SystemExit):  # fmt: skip
            print("\nExiting live training loop.")


@app.command()
def once(config: str = "mountain.toml"):
    """Performs a single capture and training cycle and then exits."""
    trainer = Trainer(config)
    trainer.run_single_cycle()


@app.command()
def live(config: str = "mountain.toml"):
    """Runs a continuous loop capturing images and weather data to train the model."""
    trainer = Trainer(config)
    trainer.live_training_loop()


@app.command()
def batch(
    folder: str | None = typer.Argument(None),
    label: int | None = None,
    config: str = "mountain.toml",
    epochs: int = 5,
    fresh: bool = False,
    labels: str | None = typer.Option(
        None, "--labels", help="Path to labels.yaml (overrides folder/labels.yaml)"
    ),
    json_summary: str | None = typer.Option(
        None,
        "--json-summary",
        help="Write a machine-readable run summary (metrics, uploaded checkpoint "
        "keys, ok/empty/no-improvement status) to this path — for unattended runs.",
    ),
):
    """Runs training using a labels index. Pass --labels path/to/labels.yaml or a folder containing labels.yaml."""
    from datetime import UTC, datetime

    started_at = datetime.now(UTC)
    per_epoch: list[dict] = []
    uploaded_keys: list[str] = []
    best_epoch: int | None = None

    def _write_summary(status: str, **extra) -> None:
        """Atomic (tmp+rename) summary write; a scheduled wrapper parses this
        instead of stdout — exit codes and prints here have known lie modes."""
        if not json_summary:
            return
        summary = {
            "status": status,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "epochs": epochs,
            "per_epoch": per_epoch,
            "best_epoch": best_epoch,
            "checkpoint_keys_uploaded": uploaded_keys,
            **extra,
        }
        out = Path(json_summary)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=2)
        tmp.rename(out)

    trainer = Trainer(config, fresh=fresh)

    if labels:
        labels_file = Path(labels)
        data_root = labels_file.parent
    elif folder:
        data_root = Path(folder)
        labels_file = data_root / "labels.yaml"
    else:
        typer.echo("Error: provide a --labels file or a folder argument.", err=True)
        raise typer.Exit(1)

    import cv2
    import numpy as np
    import yaml
    from metar import Metar
    from torchvision import transforms

    # Resolve storage backend (local or R2 with cache)
    storage = trainer.config_loader.get_storage(str(data_root))

    # Strategy: Use labels.yaml if exists, otherwise fallback to folder-wide label
    labels_map = {}
    if labels_file.exists():
        with open(labels_file, "r") as f:
            labels_map = yaml.safe_load(f) or {}
        print(f"Loaded {len(labels_map)} labels from {labels_file}")

    # Strategy: Oversample minority classes (Full = 1, Partial = 2)
    labels_full = {path: label for path, label in labels_map.items() if label == 1}
    labels_partial = {path: label for path, label in labels_map.items() if label == 2}
    labels_not_out = {path: label for path, label in labels_map.items() if label == 0}

    print(
        f"Dataset stats: {len(labels_not_out)} Not Out, {len(labels_full)} Full, {len(labels_partial)} Partial"
    )

    # We want a reasonable representation of all visible mountain frames
    # Target: 1:2:2 ratio (NotOut : Full : Partial) or similar
    max_not_out = len(labels_not_out)

    final_training_list = []
    # Add all 'Not Out' once
    for p, label in labels_not_out.items():
        final_training_list.append((p, label))

    # Oversample 'Full'
    if labels_full:
        full_factor = max(1, max_not_out // (len(labels_full) * 2))
        for p, label in labels_full.items():
            for _ in range(full_factor):
                final_training_list.append((p, label))
        print(f"  Oversampling 'Full' by {full_factor}x")

    # Oversample 'Partial'
    if labels_partial:
        partial_factor = max(1, max_not_out // (len(labels_partial) * 2))
        for p, label in labels_partial.items():
            for _ in range(partial_factor):
                final_training_list.append((p, label))
        print(f"  Oversampling 'Partial' by {partial_factor}x")

    import random

    random.shuffle(final_training_list)

    print(f"Final training set size: {len(final_training_list)} samples.")

    if not final_training_list:
        # Historically this fell through to a 0-batch "success" (exit 0,
        # best val_loss=inf, no checkpoint). Fail loudly instead.
        _write_summary(
            "empty",
            labels_loaded=len(labels_map),
            class_counts={
                "not_out": len(labels_not_out),
                "full": len(labels_full),
                "partial": len(labels_partial),
            },
        )
        typer.echo("Error: no labeled samples to train on.", err=True)
        raise typer.Exit(1)

    # Prefetch all needed files from R2 into local cache (no-op for LocalStorage)
    from collect.storage import CachedR2Storage

    if isinstance(storage, CachedR2Storage):
        prefetch_keys = []
        for rel_path, _ in final_training_list:
            prefetch_keys.append(rel_path)
            # Also add METAR file keys (try all fallback patterns)
            img_p = Path(rel_path)
            prefetch_keys.append(
                str(img_p.parent.parent / "metar" / f"{img_p.stem}.txt")
            )
            prefetch_keys.append(str(img_p.parent.parent / "metar" / "metar.txt"))
            prefetch_keys.append(str(img_p.parent / f"{img_p.stem}.txt"))
        storage.prefetch(prefetch_keys)

    batch_size = 16

    # Stratified train/val split (85/15)
    from collections import defaultdict

    by_class = defaultdict(list)
    for item in final_training_list:
        by_class[item[1]].append(item)
    train_list, val_list = [], []
    for cls, items in by_class.items():
        random.shuffle(items)
        split = max(1, int(len(items) * 0.15))
        val_list.extend(items[:split])
        train_list.extend(items[split:])
    random.shuffle(train_list)
    random.shuffle(val_list)
    print(f"Train/Val split: {len(train_list)} train, {len(val_list)} val")

    total_batches = (len(train_list) + batch_size - 1) // batch_size

    # Class weights (inverse frequency) for loss function
    from collections import Counter

    class_counts = Counter(label for _, label in train_list)
    total_samples = sum(class_counts.values())
    n_classes = 3
    class_weights = torch.tensor(
        [
            total_samples / (n_classes * class_counts.get(c, 1))
            for c in range(n_classes)
        ],
        dtype=torch.float32,
    )
    print(f"Class weights: {class_weights.tolist()}")

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    import json

    from rich.progress import (
        BarColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )

    state_file = Path("data/training_state.json")
    state_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_one(rel_path, img_label, transform_fn):
        """Load a single (image_tensor, weather_tensor, label_tensor) on CPU, or None."""
        img_rel = Path(rel_path)
        metar_candidates = [
            str(img_rel.parent.parent / "metar" / f"{img_rel.stem}.txt"),
            str(img_rel.parent.parent / "metar" / "metar.txt"),
            str(img_rel.parent / f"{img_rel.stem}.txt"),
        ]
        metar_key = next((c for c in metar_candidates if storage.exists(c)), None)
        if metar_key is None:
            return None

        try:
            img_bytes = storage.get(rel_path)
        except Exception:
            return None
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform_fn(frame_rgb)  # CPU

        metar_text = storage.get_text(metar_key).strip()
        vis, ceil = 0.0, 1.0
        try:
            obs = Metar.Metar(metar_text)
            if obs.vis:
                vis = min(obs.vis.value("SM"), 10.0) / 10.0
            if obs.sky:
                layers = [layer for layer in obs.sky if layer[0] in ["BKN", "OVC"]]
                ceil = (
                    min(layers[0][1].value("FT"), 10000.0) / 10000.0 if layers else 1.0
                )
        except Exception:
            pass

        return (
            tensor,
            torch.tensor([vis, ceil], dtype=torch.float32),
            torch.tensor(img_label),
        )

    def _run_validation():
        """Run validation pass batch-by-batch, return average loss and accuracy."""
        trainer.model_wrapper.model_dict.eval()
        cw = class_weights.to(trainer.device)
        total_loss, correct, total = 0.0, 0, 0
        buf_img, buf_w, buf_l = [], [], []

        def _flush():
            nonlocal total_loss, correct, total
            if not buf_img:
                return
            ib = torch.stack(buf_img).to(trainer.device)
            wb = torch.stack(buf_w).to(trainer.device)
            lb = torch.stack(buf_l).to(trainer.device)
            outputs = trainer.model_wrapper(ib, wb)
            total_loss += torch.nn.functional.cross_entropy(
                outputs, lb, weight=cw
            ).item() * lb.size(0)
            correct += (outputs.argmax(1) == lb).sum().item()
            total += lb.size(0)
            del ib, wb, lb, outputs
            if trainer.device == "mps":
                torch.mps.empty_cache()

        with torch.no_grad():
            for rel_path, img_label in val_list:
                item = _load_one(rel_path, img_label, val_transform)
                if item is None:
                    continue
                buf_img.append(item[0])
                buf_w.append(item[1])
                buf_l.append(item[2])
                if len(buf_img) >= batch_size:
                    _flush()
                    buf_img, buf_w, buf_l = [], [], []
            _flush()
        if total == 0:
            return float("nan"), 0.0
        return total_loss / total, correct / total

    best_val_loss = float("inf")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        epoch_task = progress.add_task("[green]Epochs...", total=epochs)

        for epoch in range(epochs):
            random.shuffle(train_list)
            batch_task = progress.add_task(
                f"[cyan]Epoch {epoch + 1}/{epochs}...", total=total_batches
            )

            image_list, weather_list, label_list = [], [], []
            epoch_losses = []
            batches_complete = 0

            for rel_path, img_label in train_list:
                # Resolve METAR key via fallback pattern (via storage so R2 works when active)
                img_rel = Path(rel_path)
                metar_candidates = [
                    str(img_rel.parent.parent / "metar" / f"{img_rel.stem}.txt"),
                    str(img_rel.parent.parent / "metar" / "metar.txt"),
                    str(img_rel.parent / f"{img_rel.stem}.txt"),
                ]
                metar_key = next(
                    (c for c in metar_candidates if storage.exists(c)), None
                )
                if metar_key is None:
                    continue

                try:
                    img_bytes = storage.get(rel_path)
                except Exception:
                    continue
                frame = cv2.imdecode(
                    np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR
                )
                if frame is None:
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = train_transform(frame_rgb).to(trainer.device)

                metar_text = storage.get_text(metar_key).strip()
                vis, ceil = 0.0, 1.0
                try:
                    obs = Metar.Metar(metar_text)
                    if obs.vis:
                        vis = min(obs.vis.value("SM"), 10.0) / 10.0
                    if obs.sky:
                        layers = [
                            layer for layer in obs.sky if layer[0] in ["BKN", "OVC"]
                        ]
                        ceil = (
                            min(layers[0][1].value("FT"), 10000.0) / 10000.0
                            if layers
                            else 1.0
                        )
                except Exception:
                    pass

                image_list.append(tensor)
                weather_list.append(torch.tensor([vis, ceil], dtype=torch.float32))
                label_list.append(torch.tensor(img_label))

                if len(image_list) >= batch_size:
                    loss = trainer.model_wrapper.train_step(
                        torch.stack(image_list),
                        torch.stack(weather_list),
                        torch.stack(label_list),
                        trainer.optimizer,
                        class_weights=class_weights,
                    )
                    epoch_losses.append(loss)
                    image_list, weather_list, label_list = [], [], []

                    batches_complete += 1
                    progress.update(batch_task, advance=1)

                    avg_loss = sum(epoch_losses) / len(epoch_losses)
                    progress.update(
                        batch_task,
                        description=f"[cyan]Epoch {epoch + 1}/{epochs} (Loss: {avg_loss:.4f})",
                    )

                    state = {
                        "status": "running",
                        "epoch": epoch + 1,
                        "total_epochs": epochs,
                        "batches_complete": batches_complete,
                        "total_batches": total_batches,
                        "current_loss": avg_loss,
                    }
                    tmp_state = state_file.with_suffix(".tmp")
                    with open(tmp_state, "w") as f:
                        json.dump(state, f)
                    tmp_state.rename(state_file)

            if image_list:
                loss = trainer.model_wrapper.train_step(
                    torch.stack(image_list),
                    torch.stack(weather_list),
                    torch.stack(label_list),
                    trainer.optimizer,
                    class_weights=class_weights,
                )
                epoch_losses.append(loss)
                batches_complete += 1
                progress.update(batch_task, advance=1)
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                progress.update(
                    batch_task,
                    description=f"[cyan]Epoch {epoch + 1}/{epochs} (Loss: {avg_loss:.4f})",
                )

            avg_loss = (
                sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
            )

            # Validation
            val_loss, val_acc = _run_validation()
            progress.remove_task(batch_task)
            progress.update(
                epoch_task,
                advance=1,
                description=f"[green]Epoch {epoch + 1}: train={avg_loss:.4f} val={val_loss:.4f} acc={val_acc:.1%}",
            )
            print(
                f"  Epoch {epoch + 1}: train_loss={avg_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.1%}"
            )
            per_epoch.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                uploaded_keys = trainer.model_wrapper.save_checkpoint(
                    trainer.config_loader.checkpoint_dir, storage=storage
                )
                print(f"  ↳ Best model saved (val_loss={val_loss:.4f})")

    if state_file.exists():
        state = {
            "status": "complete",
            "epoch": epochs,
            "total_epochs": epochs,
            "batches_complete": total_batches,
            "total_batches": total_batches,
            "current_loss": avg_loss if "avg_loss" in locals() else 0.0,
        }
        with open(state_file.with_suffix(".tmp"), "w") as f:
            json.dump(state, f)
        state_file.with_suffix(".tmp").rename(state_file)

    # Reload the best checkpoint (saved during training) for evaluation
    trainer.model_wrapper.load_checkpoint(
        trainer.config_loader.checkpoint_dir, storage=storage
    )

    # Clean up R2 cache if used
    if isinstance(storage, CachedR2Storage):
        storage.clear_cache()

    if best_val_loss == float("inf"):
        _write_summary("no-improvement", labels_loaded=len(labels_map))
        typer.echo(
            "Error: validation never produced a finite loss; no checkpoint saved.",
            err=True,
        )
        raise typer.Exit(1)

    _write_summary(
        "ok",
        labels_loaded=len(labels_map),
        class_counts={
            "not_out": len(labels_not_out),
            "full": len(labels_full),
            "partial": len(labels_partial),
        },
        train_n=len(train_list),
        val_n=len(val_list),
        best_val_loss=best_val_loss,
    )

    print(
        f"\nTraining complete (best val_loss={best_val_loss:.4f}). Running final evaluation..."
    )
    import sys

    sys.path.append(str(Path.cwd()))
    from tools.evaluate import evaluate

    evaluate(trainer.config_loader.checkpoint_dir, str(labels_file))


@app.command()
def schedule(config: str = "mountain.toml"):
    """Installs the launchctl service for periodic training."""
    trainer = Trainer(config)
    config_loader = trainer.config_loader
    current_dir = Path.cwd().absolute()
    executable = subprocess.check_output(["which", "uv"], text=True).strip()
    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.mountain.trainer.plist"

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.mountain.trainer</string>
    <key>ProgramArguments</key>
    <array>
        <string>{executable}</string>
        <string>run</string>
        <string>training</string>
        <string>once</string>
        <string>--config</string>
        <string>{current_dir / config}</string>
    </array>
    <key>StartInterval</key>
    <integer>{config_loader.schedule_seconds}</integer>
    <key>StandardErrorPath</key>
    <string>/tmp/mountain_trainer.err</string>
    <key>StandardOutPath</key>
    <string>/tmp/mountain_trainer.out</string>
    <key>WorkingDirectory</key>
    <string>{current_dir}</string>
</dict>
</plist>
"""
    with open(plist_path, "w") as f:
        f.write(plist_content)
    subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
    subprocess.run(["launchctl", "load", str(plist_path)])
    print(
        f"Service installed at {plist_path}. Interval: {config_loader.schedule_seconds}s"
    )


@app.command()
def unschedule():
    """Unloads and removes the launchctl service."""
    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.mountain.trainer.plist"
    if plist_path.exists():
        subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
        plist_path.unlink()
        print("Service removed.")
    else:
        print("Service not found.")


if __name__ == "__main__":
    app()

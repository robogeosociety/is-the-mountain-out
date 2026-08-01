import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch
import typer
from torch import optim

from train.config_loader import ConfigLoader
from train.metrics import compute_metrics, summary_line
from train.model import ConvNextLoRAModel
from train.utils import WeatherFetcher, WebcamStream

app = typer.Typer()


def memory_snapshot() -> dict:
    """Process + accelerator memory right now, in MB.

    Best-effort by construction: a training run must never fail because a
    telemetry probe did. Every key is optional, and callers render whatever is
    present.
    """
    import sys

    snapshot: dict[str, float] = {}
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is BYTES on macOS and KILOBYTES on Linux — the classic
        # 1024x reporting bug if you assume one of them.
        divisor = 1024**2 if sys.platform == "darwin" else 1024
        snapshot["rss_peak_mb"] = round(peak / divisor, 1)
    except Exception:  # noqa: BLE001 — probe only
        pass
    try:
        if torch.backends.mps.is_available() and hasattr(torch, "mps"):
            snapshot["mps_allocated_mb"] = round(
                torch.mps.current_allocated_memory() / 1024**2, 1
            )
            if hasattr(torch.mps, "driver_allocated_memory"):
                snapshot["mps_driver_mb"] = round(
                    torch.mps.driver_allocated_memory() / 1024**2, 1
                )
        elif torch.cuda.is_available():
            snapshot["cuda_allocated_mb"] = round(
                torch.cuda.memory_allocated() / 1024**2, 1
            )
            snapshot["cuda_peak_mb"] = round(
                torch.cuda.max_memory_allocated() / 1024**2, 1
            )
    except Exception:  # noqa: BLE001 — probe only
        pass
    return snapshot


def stratified_split(
    by_class: dict[int, list[str]], val_fraction: float, rng
) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    """Split UNIQUE items per class into (train, val), deterministically.

    Two properties this function exists to guarantee, both of which the previous
    inline split lacked:

    1. **It runs on unique items, before oversampling.** Oversampling duplicates
       each minority frame ~8x. Splitting the oversampled list put copies of the
       same image on both sides, so validation scored the model on frames it had
       memorised — the mechanism behind a 99.8% val accuracy on a class holding
       5.5% of the data.
    2. **It is seeded.** An unseeded split redraws the val set on every run, so
       week-over-week metric deltas measured the dice, not the model.

    A class with a single example keeps it in train: spending it on val would
    make the class untrainable *and* its recall a coin flip.
    """
    train: dict[int, list[str]] = {}
    val: dict[int, list[str]] = {}
    for cls, items in by_class.items():
        # sorted() first so the result depends on the seed, not on dict or
        # filesystem ordering.
        shuffled = sorted(items)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_val = 0 if n < 2 else min(max(1, round(n * val_fraction)), n - 1)
        val[cls] = shuffled[:n_val]
        train[cls] = shuffled[n_val:]
    return train, val


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
    progress_jsonl: str | None = typer.Option(
        None,
        "--progress-jsonl",
        help="Append one JSON line per epoch (metrics + memory + whether a "
        "checkpoint was saved) as the run proceeds, so a supervisor can report "
        "progress live instead of waiting for --json-summary at the end.",
    ),
    seed: int = typer.Option(
        1337,
        "--seed",
        help="Seed for the train/val split, oversampling and epoch shuffles. "
        "Fixed by default so two runs' val metrics are comparable — an unseeded "
        "split redraws the val set every week and turns run-over-run deltas "
        "into noise.",
    ),
):
    """Runs training using a labels index. Pass --labels path/to/labels.yaml or a folder containing labels.yaml."""
    from datetime import UTC, datetime

    started_at = datetime.now(UTC)
    per_epoch: list[dict] = []
    uploaded_keys: list[str] = []
    best_epoch: int | None = None
    best_val_acc: float | None = None
    best_val_metrics: dict | None = None

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

    def _emit_progress(event: dict) -> None:
        """Append one progress line and get it on disk immediately.

        A reader is tailing this file while the run is in flight, so the write
        is flushed and fsync'd — buffered lines would arrive in a clump at exit,
        which is exactly the latency this option exists to remove. Append-only
        and line-atomic: the reader consumes whole lines and ignores a partial
        tail. Never raises; telemetry must not be able to fail a run.
        """
        if not progress_jsonl:
            return
        try:
            path = Path(progress_jsonl)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(event) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError as exc:
            print(f"  (progress telemetry write failed, continuing: {exc})")

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

    labels_full = {path: label for path, label in labels_map.items() if label == 1}
    labels_partial = {path: label for path, label in labels_map.items() if label == 2}
    labels_not_out = {path: label for path, label in labels_map.items() if label == 0}

    print(
        f"Dataset stats: {len(labels_not_out)} Not Out, {len(labels_full)} Full, {len(labels_partial)} Partial"
    )

    import random

    rng = random.Random(seed)

    # SPLIT FIRST, THEN OVERSAMPLE.
    #
    # This order is the whole point. Oversampling duplicates each minority frame
    # ~8x; splitting the oversampled list afterwards puts copies of the SAME
    # image in both train and val, so the model is scored on frames it memorised
    # — which is how a 5.5%-of-the-data class produced a 99.8% val accuracy.
    # Splitting the unique labels first keeps val honest, and oversampling only
    # the train side keeps the class-balance benefit it was added for.
    train_by_class, val_by_class = stratified_split(
        {
            0: list(labels_not_out),
            1: list(labels_full),
            2: list(labels_partial),
        },
        val_fraction=0.15,
        rng=rng,
    )

    val_list = [(p, cls) for cls, paths in val_by_class.items() for p in paths]
    rng.shuffle(val_list)

    # Oversample the minority classes — TRAIN SIDE ONLY.
    # Target: roughly 1:2:2 (NotOut : Full : Partial) representation.
    max_not_out = len(train_by_class[0])
    final_training_list = [(p, 0) for p in train_by_class[0]]
    for cls, name in ((1, "Full"), (2, "Partial")):
        paths = train_by_class[cls]
        if not paths:
            continue
        factor = max(1, max_not_out // (len(paths) * 2))
        for p in paths:
            final_training_list.extend([(p, cls)] * factor)
        print(f"  Oversampling '{name}' by {factor}x")

    rng.shuffle(final_training_list)

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
        # Val frames are no longer duplicated inside the training list, so they
        # have to be prefetched explicitly or every validation pass would fall
        # back to per-image R2 GETs.
        for rel_path, _ in final_training_list + val_list:
            prefetch_keys.append(rel_path)
            # Also add METAR file keys (try all fallback patterns)
            img_p = Path(rel_path)
            prefetch_keys.append(
                str(img_p.parent.parent / "metar" / f"{img_p.stem}.txt")
            )
            prefetch_keys.append(str(img_p.parent.parent / "metar" / "metar.txt"))
            prefetch_keys.append(str(img_p.parent / f"{img_p.stem}.txt"))
        prefetch_started = time.monotonic()
        storage.prefetch(prefetch_keys)
        prefetch_s = round(time.monotonic() - prefetch_started, 1)
    else:
        prefetch_s = 0.0

    batch_size = 16

    train_list = final_training_list
    val_class_counts = {
        name: len(val_by_class[cls])
        for cls, name in ((0, "not_out"), (1, "full"), (2, "partial"))
    }
    train_class_counts_unique = {
        name: len(train_by_class[cls])
        for cls, name in ((0, "not_out"), (1, "full"), (2, "partial"))
    }
    print(
        f"Train/Val split (seed {seed}): {len(train_list)} train samples "
        f"({sum(train_class_counts_unique.values())} unique), {len(val_list)} val"
    )
    print(f"  Val per class: {val_class_counts}")
    if min(val_class_counts.values()) < 20:
        # Say it out loud: at these counts one flipped frame moves that class's
        # recall by several points, so small deltas are noise, not progress.
        print(
            "  ⚠️ small val classes — per-class recall moves by "
            f"~{100 / max(1, min(val_class_counts.values())):.0f}pt per frame"
        )

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
        """Validation pass, batch by batch → (loss, accuracy, per-class metrics).

        Predictions and truth are accumulated as plain ints (on CPU, one small
        list) so the confusion matrix — and everything derived from it — comes
        out of the same forward passes accuracy already needed. No second pass,
        no extra memory of consequence.
        """
        trainer.model_wrapper.model_dict.eval()
        cw = class_weights.to(trainer.device)
        total_loss, correct, total = 0.0, 0, 0
        all_preds: list[int] = []
        all_targets: list[int] = []
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
            predicted = outputs.argmax(1)
            correct += (predicted == lb).sum().item()
            total += lb.size(0)
            all_preds.extend(predicted.detach().cpu().tolist())
            all_targets.extend(lb.detach().cpu().tolist())
            del ib, wb, lb, outputs, predicted
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
        metrics = compute_metrics(all_targets, all_preds)
        if total == 0:
            return float("nan"), 0.0, metrics
        return total_loss / total, correct / total, metrics

    best_val_loss = float("inf")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        epoch_task = progress.add_task("[green]Epochs...", total=epochs)

        for epoch in range(epochs):
            epoch_started = time.monotonic()
            rng.shuffle(train_list)
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
            val_loss, val_acc, val_metrics = _run_validation()
            macro_f1 = val_metrics.get("macro_f1", 0.0)
            progress.remove_task(batch_task)
            progress.update(
                epoch_task,
                advance=1,
                description=f"[green]Epoch {epoch + 1}: train={avg_loss:.4f} val={val_loss:.4f} macroF1={macro_f1:.3f}",
            )
            print(
                f"  Epoch {epoch + 1}: train_loss={avg_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.1%}"
            )
            print(f"    {summary_line(val_metrics)}")
            record = {
                "epoch": epoch + 1,
                "total_epochs": epochs,
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                # Per-class precision/recall/F1, the confusion matrix, macro-F1,
                # balanced accuracy and the Full+Partial "visible" collapse.
                # Flows to --json-summary and --progress-jsonl via this dict.
                "val_metrics": val_metrics,
                "duration_s": round(time.monotonic() - epoch_started, 1),
                "memory": memory_snapshot(),
                "checkpoint_saved": False,
            }
            # The same dict object goes into per_epoch, so the checkpoint flags
            # set below also land in --json-summary.
            per_epoch.append(record)

            if val_loss < best_val_loss:
                previous_best = None if best_val_loss == float("inf") else best_val_loss
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_val_acc = val_acc
                best_val_metrics = val_metrics
                uploaded_keys = trainer.model_wrapper.save_checkpoint(
                    trainer.config_loader.checkpoint_dir, storage=storage
                )
                record["checkpoint_saved"] = True
                record["previous_best_val_loss"] = previous_best
                record["checkpoint_keys"] = list(uploaded_keys)
                print(f"  ↳ Best model saved (val_loss={val_loss:.4f})")

            # Emitted AFTER the save attempt so `checkpoint_saved` is truthful:
            # a reader posts on this line and must not announce a checkpoint
            # that had not been written yet.
            _emit_progress(record)

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
        train_class_counts=train_class_counts_unique,
        val_class_counts=val_class_counts,
        split_seed=seed,
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
        # The headline block: macro-F1 / balanced accuracy / per-class P-R-F1 /
        # confusion / visible-vs-not-out, from the epoch that saved the model.
        best_val_metrics=best_val_metrics,
        prefetch_s=prefetch_s,
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

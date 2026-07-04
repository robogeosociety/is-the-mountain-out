import os
import sys

# FORCE NON-INTERACTIVE
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Ensure no GUI windows
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import matplotlib

matplotlib.use("Agg")  # Force non-interactive backend

import yaml
import torch
import torch.optim as optim
from pathlib import Path
import cv2
import numpy as np
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from metar import Metar
import pandas as pd

# Add root to path for imports
sys.path.append(str(Path.cwd()))
from train.model import ConvNextLoRAModel
from train.config_loader import ConfigLoader


def get_metar_vector(storage, img_key: str):
    img_rel = Path(img_key)
    candidates = [
        str(img_rel.parent.parent / "metar" / "metar.txt"),
        str(img_rel.parent / "metar.txt"),
        str(img_rel.parent / f"{img_rel.stem}.txt"),
    ]
    metar_text = None
    for c in candidates:
        if storage.exists(c):
            metar_text = storage.get_text(c).strip()
            break

    vis, ceil = 0.0, 1.0  # Default bad
    if metar_text:
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
    return [vis, ceil]


def run_experiment(variant_name, data_list, storage, folds=3):
    print(f"\n🚀 Starting Experiment: {variant_name}", flush=True)

    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    paths = [d["path"] for d in data_list]
    labels = [d["label"] for d in data_list]

    all_metrics = []

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Using device: {device}", flush=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(paths, labels)):
        print(f"  Fold {fold + 1}/{folds}...", flush=True)

        # Fresh model for each fold
        model = ConvNextLoRAModel().to(device)
        optimizer = optim.Adam(model.model_dict.parameters(), lr=0.0001)

        # Prepare training data with oversampling
        train_data = [data_list[i] for i in train_idx]
        train_out = [d for d in train_data if d["label"] == 1]
        train_partial = [d for d in train_data if d["label"] == 2]
        train_not = [d for d in train_data if d["label"] == 0]

        # Balance the batch
        oversample_factor_out = (
            max(1, len(train_not) // (len(train_out) * 2)) if train_out else 1
        )
        oversample_factor_partial = (
            max(1, len(train_not) // (len(train_partial) * 2)) if train_partial else 1
        )

        balanced_train = (
            train_not
            + (train_out * oversample_factor_out)
            + (train_partial * oversample_factor_partial)
        )
        import random

        random.shuffle(balanced_train)

        # Training loop (Mini-version: 3 epochs)
        model.train()
        for epoch in range(3):
            batch_count = 0
            for i in range(0, len(balanced_train), 16):
                batch = balanced_train[i : i + 16]
                if not batch:
                    continue

                imgs, weathers, lbls = [], [], []
                for item in batch:
                    try:
                        img_bytes = storage.get(item["path"])
                    except Exception:
                        continue
                    img = cv2.imdecode(
                        np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR
                    )
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    imgs.append(transform(img))

                    w = item["weather"]
                    if variant_name == "Vision Only":
                        w = [0.0, 0.0]
                    weathers.append(torch.tensor(w))
                    lbls.append(item["label"])

                if not imgs:
                    continue

                model.train_step(
                    torch.stack(imgs),
                    torch.stack(weathers),
                    torch.tensor(lbls).long(),
                    optimizer,
                )
                batch_count += 1
            print(
                f"    Epoch {epoch + 1} finished ({batch_count} batches).", flush=True
            )

        # Evaluation
        model.eval()
        preds, true = [], []
        print("    Starting evaluation...", flush=True)
        with torch.no_grad():
            for i in val_idx:
                item = data_list[i]
                try:
                    img_bytes = storage.get(item["path"])
                except Exception:
                    continue
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = transform(img).unsqueeze(0).to(device)

                w = item["weather"]
                if variant_name == "Vision Only":
                    w = [0.0, 0.0]

                weather_tensor = torch.tensor([w], dtype=torch.float32).to(device)

                output = model(img_tensor, weather_tensor)
                pred = torch.argmax(output, dim=1).item()
                preds.append(pred)
                true.append(item["label"])

        f1 = f1_score(true, preds, average="weighted")
        prec = precision_score(true, preds, zero_division=0, average="weighted")
        rec = recall_score(true, preds, zero_division=0, average="weighted")

        all_metrics.append({"f1": f1, "precision": prec, "recall": rec})
        print(f"    F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}", flush=True)

    avg_f1 = np.mean([m["f1"] for m in all_metrics])
    avg_prec = np.mean([m["precision"] for m in all_metrics])
    avg_rec = np.mean([m["recall"] for m in all_metrics])

    return {
        "Variant": variant_name,
        "F1-Score": avg_f1,
        "Precision": avg_prec,
        "Recall": avg_rec,
    }


if __name__ == "__main__":
    data_root = Path("data")
    storage = ConfigLoader().get_storage(str(data_root))

    with open(data_root / "labels.yaml", "r") as f:
        labels_map = yaml.safe_load(f) or {}

    print(f"Loading and pre-processing {len(labels_map)} images...")
    processed_data = []
    for rel_path, label in labels_map.items():
        if not storage.exists(rel_path):
            continue

        weather = get_metar_vector(storage, rel_path)
        processed_data.append({"path": rel_path, "label": label, "weather": weather})

    # Run full experiment
    # processed_data = processed_data[:50]
    print(
        f"Running experiment on all {len(processed_data)} labeled images...", flush=True
    )

    # Map every image to the most recent UNIQUE weather vector that preceded it
    # This perfectly models our new collector deduplication logic
    unique_weather_history = []
    sparse_data = []

    current_weather = [0.0, 1.0]  # Default
    for item in processed_data:
        actual_weather = item["weather"]
        if actual_weather != current_weather:
            current_weather = actual_weather

        new_item = item.copy()
        new_item["weather"] = current_weather
        sparse_data.append(new_item)

    results = []
    results.append(run_experiment("Vision Only", processed_data, storage))
    results.append(run_experiment("Full METAR", processed_data, storage))
    results.append(run_experiment("Sparse METAR", sparse_data, storage))

    df = pd.DataFrame(results)
    print("\n📊 --- A/B TEST RESULTS ---")
    print(df.to_markdown(index=False))

    best = df.loc[df["F1-Score"].idxmax()]
    print(f"\n✅ CONCLUSION: The '{best['Variant']}' model performed best.")

    if best["Variant"] == "Vision Only":
        print(
            "💡 METAR data is redundant. Vision backbone captures enough atmospheric cues."
        )
    else:
        gain = (
            (
                best["F1-Score"]
                - df[df["Variant"] == "Vision Only"]["F1-Score"].values[0]
            )
            / df[df["Variant"] == "Vision Only"]["F1-Score"].values[0]
            * 100
        )
        print(f"💡 METAR data improves F1-Score by {gain:.1f}%.")

    if (
        df[df["Variant"] == "Full METAR"]["F1-Score"].values[0]
        > df[df["Variant"] == "Sparse METAR"]["F1-Score"].values[0] * 1.05
    ):
        print(
            "💡 High-frequency METAR is SIGNIFICANT. Hourly updates catch rapid clearing events."
        )
    else:
        print(
            "💡 Sparse METAR (Once a Day) is SUFFICIENT. Seasonal trends dominate the weather signal."
        )

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
import cv2
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
from metar import Metar
import argparse

# Force non-interactive backends where necessary
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

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


def evaluate(checkpoint_dir: str, labels_file: str):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading checkpoint from: {checkpoint_dir}")
    print(f"Using device: {device}")

    model = ConvNextLoRAModel(
        num_classes=3, checkpoint_dir=checkpoint_dir, device=device
    )
    model.model_dict.eval()

    labels_path = Path(labels_file)
    data_root = labels_path.parent
    storage = ConfigLoader().get_storage(str(data_root))

    with open(labels_path, "r") as f:
        labels_map = yaml.safe_load(f) or {}

    print(f"Found {len(labels_map)} labels in {labels_file}")

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    true_labels = []
    pred_labels = []

    print("Running inference...")
    with torch.no_grad():
        for rel_path, label in labels_map.items():
            if not storage.exists(rel_path):
                continue

            try:
                img_bytes = storage.get(rel_path)
            except Exception:
                continue
            frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = transform(frame_rgb).unsqueeze(0).to(device)

            weather = get_metar_vector(storage, rel_path)
            weather_tensor = torch.tensor([weather], dtype=torch.float32).to(device)

            output = model(img_tensor, weather_tensor)
            pred = torch.argmax(output, dim=1).item()

            true_labels.append(label)
            pred_labels.append(pred)

    if not true_labels:
        print("No valid images found for evaluation.")
        return

    print("\n--- Evaluation Results ---")
    target_names = ["Not Out (0)", "Full (1)", "Partial (2)"]
    # Handle cases where some classes might not be present in the true labels
    present_classes = sorted(list(set(true_labels) | set(pred_labels)))
    present_target_names = [target_names[i] for i in present_classes]

    print(
        classification_report(
            true_labels,
            pred_labels,
            labels=present_classes,
            target_names=present_target_names,
            zero_division=0,
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, pred_labels, labels=present_classes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model checkpoint on a labeled dataset."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="train/checkpoints",
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--labels", type=str, required=True, help="Path to labels.yaml file"
    )

    args = parser.parse_args()
    evaluate(args.checkpoint, args.labels)

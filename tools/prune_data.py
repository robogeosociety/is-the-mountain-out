import io
import sys
import argparse
import yaml
from pathlib import Path
from PIL import Image, ImageChops, ImageStat
from metar import Metar
from datetime import datetime, UTC

sys.path.append(str(Path.cwd()))
from train.config_loader import ConfigLoader


def load_labels(data_root):
    labels_path = Path(data_root) / "labels.yaml"
    if labels_path.exists():
        with open(labels_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_labels(data_root, labels, storage):
    # Mirror to both local and storage so subsequent runs see the new labels
    # whether reading via storage.get_text or the local file directly.
    labels_path = Path(data_root) / "labels.yaml"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(labels)
    with open(labels_path, "w") as f:
        f.write(text)
    storage.put_text("labels.yaml", text)


def get_metar_data(storage, img_key: str):
    img_rel = Path(img_key)
    candidates = [
        str(img_rel.parent.parent / "metar" / "metar.txt"),
        str(img_rel.parent / "metar.txt"),
        str(img_rel.parent / f"{img_rel.stem}.txt"),
    ]
    for c in candidates:
        if storage.exists(c):
            return storage.get_text(c).strip()
    return None


def _capture_dir_for(img_key: str) -> str:
    # data layout: <date>/<HHMMSS_us_UTC>/images/<...>.jpg
    parts = Path(img_key).parts
    if len(parts) < 3:
        return ""
    return f"{parts[0]}/{parts[1]}"


def _mtime_from_key(img_key: str) -> float:
    # Keys are YYYYMMDD/HHMMSS_us_UTC/images/...jpg — the directory name IS the
    # capture timestamp, which is reliable across local and R2 (R2 list_keys
    # gives no mtime through the storage protocol).
    parts = Path(img_key).parts
    date_str, time_str = parts[0], parts[1]  # 20260222, 220834_724181_UTC
    hhmmss = time_str[:6]
    return (
        datetime.strptime(f"{date_str} {hhmmss}", "%Y%m%d %H%M%S")
        .replace(tzinfo=UTC)
        .timestamp()
    )


def _open_pil(storage, img_key: str):
    return Image.open(io.BytesIO(storage.get(img_key)))


def prune_dataset(
    data_root="data",
    min_seconds=300,
    dark_thresh=10.0,
    diff_thresh=2.0,
    dry_run=True,
    force_keep_hourly=True,
    auto_label_metar=True,
):
    storage = ConfigLoader().get_storage(data_root)
    labels = load_labels(data_root)

    image_keys = [
        k for k in storage.list_keys("") if k.endswith(".jpg") and k not in labels
    ]
    image_keys.sort()  # Lexicographic == chronological for our key layout
    images = [(_mtime_from_key(k), k) for k in image_keys]

    print(f"🔍 Found {len(images)} unlabeled images in '{data_root}'")

    to_delete_dirs = set()
    reasons = {"time": 0, "dark": 0, "duplicate": 0, "metar_auto": 0}

    last_kept_time = 0
    last_kept_img = None
    last_forced_hour = -1

    for mtime, img_key in images:
        capture_dir = _capture_dir_for(img_key)

        # 1. METAR Auto-Labeling (Obvious "Not Out")
        if auto_label_metar:
            metar_text = get_metar_data(storage, img_key)
            if metar_text:
                try:
                    obs = Metar.Metar(metar_text)
                    vis = obs.vis.value("SM") if obs.vis else 10.0
                    ceil = 10000.0
                    if obs.sky:
                        layers = [
                            layer for layer in obs.sky if layer[0] in ["BKN", "OVC"]
                        ]
                        if layers:
                            ceil = layers[0][1].value("FT")

                    # Vis crap OR ceiling below Rainier's peak area (~8000ft)
                    if vis < 3.0 or ceil < 6000:
                        if not dry_run:
                            labels[img_key] = 0
                        reasons["metar_auto"] += 1
                        continue  # Auto-label, do not delete
                except Exception:
                    pass

        # 2. Force Keep Logic (1 per hour for darkness baseline)
        dt = datetime.fromtimestamp(mtime, UTC)
        hour_key = (dt.day, dt.hour)
        if force_keep_hourly and hour_key != last_forced_hour:
            last_forced_hour = hour_key
            last_kept_time = mtime
            try:
                with _open_pil(storage, img_key) as img:
                    last_kept_img = img.convert("L").copy()
                continue
            except Exception:
                pass

        # 3. Temporal Pruning
        if mtime - last_kept_time < min_seconds:
            to_delete_dirs.add(capture_dir)
            reasons["time"] += 1
            continue

        try:
            with _open_pil(storage, img_key) as img:
                img_gray = img.convert("L")
                stat = ImageStat.Stat(img_gray)
                avg_brightness = stat.mean[0]

                # 4. Darkness Pruning
                if avg_brightness < dark_thresh:
                    to_delete_dirs.add(capture_dir)
                    reasons["dark"] += 1
                    continue

                # 5. Redundancy / Diff Pruning
                if last_kept_img is not None:
                    diff = ImageChops.difference(img_gray, last_kept_img)
                    diff_stat = ImageStat.Stat(diff)
                    avg_diff = diff_stat.mean[0]

                    if avg_diff < diff_thresh:
                        to_delete_dirs.add(capture_dir)
                        reasons["duplicate"] += 1
                        continue

                last_kept_time = mtime
                last_kept_img = img_gray.copy()

        except Exception as e:
            print(f"Error processing {img_key}: {e}")

    if not dry_run:
        save_labels(data_root, labels, storage)

    total_deleted = len(to_delete_dirs)

    print("\n📊 --- Pruning & Auto-Labeling Results ---")
    print(f"Auto-Labeled (METAR Low Vis/Ceiling): {reasons['metar_auto']}")
    print(f"Pruned (Time Constraints):           {reasons['time']}")
    print(f"Pruned (Too Dark):                   {reasons['dark']}")
    print(f"Pruned (No Scene Change):            {reasons['duplicate']}")
    print(f"Total Capture Dirs to Delete:        {total_deleted}")

    if dry_run:
        print("\n🛑 DRY RUN: No files deleted, no labels saved.")
        return

    print("\n🗑️ Executing deletions and saving auto-labels...")
    deleted_keys = 0
    for d in to_delete_dirs:
        prefix = d + "/"
        keys = storage.list_keys(prefix)
        for k in keys:
            try:
                storage.delete(k)
                deleted_keys += 1
            except Exception as e:
                print(f"Failed to delete {k}: {e}")
    print(f"Done. Deleted {deleted_keys} keys across {total_deleted} capture dirs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--min-sec", type=int, default=300)
    parser.add_argument("--dark", type=float, default=10.0)
    parser.add_argument("--diff", type=float, default=2.0)
    args = parser.parse_args()

    prune_dataset(
        dry_run=not args.execute,
        min_seconds=args.min_sec,
        dark_thresh=args.dark,
        diff_thresh=args.diff,
    )

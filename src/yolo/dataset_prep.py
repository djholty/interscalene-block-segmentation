import os
import shutil
import random
from pathlib import Path

def make_split_dataset(
    X_dir: str,
    Y_dir: str,
    out_dir: str = "dataset",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    copy: bool = True,  # True = copy, False = move
):
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-9, "Ratios must sum to 1.0"

    X_dir = Path(X_dir)
    Y_dir = Path(Y_dir)
    out_dir = Path(out_dir)

    img_paths = sorted(
        p for p in X_dir.glob("*.tif")
        if "mask" not in p.stem.lower())
    if not img_paths:
        raise FileNotFoundError(f"No .tif files found in {X_dir}")

    # Build list of (image_path, label_path) - only include pairs with non-empty label files
    # Skip images that don't have visible masks (empty or missing label files)
    pairs = []
    skipped_no_label = 0
    skipped_empty_label = 0
    for img_path in img_paths:
        base = img_path.stem
        label_path = Y_dir / f"{base}.txt"
        
        # Skip if label file doesn't exist
        if not label_path.exists():
            skipped_no_label += 1
            continue
        
        # Skip if label file is empty (no visible masks)
        if label_path.stat().st_size == 0:
            skipped_empty_label += 1
            continue
        
        # Only include pairs with non-empty label files (visible masks)
        pairs.append((img_path, label_path))

    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    # remainder goes to test (avoids rounding issues)
    n_test = n - n_train - n_val

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    # Create folders
    for split in splits:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    op = shutil.copy2 if copy else shutil.move

    # Copy/move files (all pairs have valid, non-empty label files)
    for split, items in splits.items():
        for img_path, label_path in items:
            # image
            dst_img = out_dir / "images" / split / img_path.name
            op(img_path, dst_img)

            # label (all labels are non-empty - only visible masks included)
            dst_label = out_dir / "labels" / split / f"{img_path.stem}.txt"
            op(label_path, dst_label)

    # Write data.yaml
    yaml_text = f"""path: {out_dir.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: target
"""
    (out_dir / "data.yaml").write_text(yaml_text)

    print(f"Total images with visible masks: {n}")
    print(f"Skipped (no label file): {skipped_no_label}")
    print(f"Skipped (empty label file - no visible masks): {skipped_empty_label}")
    for split in ["train", "val", "test"]:
        print(f"{split}: {len(splits[split])}")
    print(f"Created: {(out_dir / 'data.yaml')}")
    print("Done.")

if __name__ == "__main__":
    make_split_dataset("data_brachial_plexus", 
                       "output/yolo_labels", 
                       out_dir="src/yolo/dataset", 
                       train_ratio=0.7, 
                       val_ratio=0.2, 
                       test_ratio=0.1, 
                       seed=42, 
                       copy=True)

import os
import hashlib
from PIL import Image

# ─── CONFIG ───────────────────────────────────────────────
SOURCE = "archive\dataset_new"
TARGET = "dataset_binary"
SIZE   = (128, 128)
CLASS_MAP = {
    'Closed': 'Drowsy',
    'yawn':   'Drowsy',
    'Open':   'Alert',
    'no_yawn':'Alert'
}
# ──────────────────────────────────────────────────────────


# STEP 1 — Create binary folder structure
def create_folders():
    for split in ['train', 'test']:
        for cls in ['Drowsy', 'Alert']:
            os.makedirs(f"{TARGET}/{split}/{cls}", exist_ok=True)
    print("✔ Folders created.\n")


# STEP 2 — Remove corrupted images
def remove_corrupted():
    removed = 0
    for split in ['train', 'test']:
        for cls in os.listdir(os.path.join(SOURCE, split)):
            cls_path = os.path.join(SOURCE, split, cls)
            if not os.path.isdir(cls_path):
                continue
            for fname in os.listdir(cls_path):
                fpath = os.path.join(cls_path, fname)
                try:
                    with Image.open(fpath) as img:
                        img.verify()
                    with Image.open(fpath) as img:
                        img.convert('RGB')
                except Exception as e:
                    print(f"  Removing corrupted: {fpath} | {e}")
                    os.remove(fpath)
                    removed += 1
    print(f"✔ Corrupted images removed: {removed}\n")


# STEP 3 — Remove duplicate images
def remove_duplicates():
    seen_hashes = set()
    removed = 0
    for split in ['train', 'test']:
        for cls in os.listdir(os.path.join(SOURCE, split)):
            cls_path = os.path.join(SOURCE, split, cls)
            if not os.path.isdir(cls_path):
                continue
            for fname in os.listdir(cls_path):
                fpath = os.path.join(cls_path, fname)
                try:
                    with open(fpath, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    if file_hash in seen_hashes:
                        print(f"  Duplicate removed: {fpath}")
                        os.remove(fpath)
                        removed += 1
                    else:
                        seen_hashes.add(file_hash)
                except Exception as e:
                    print(f"  Error: {fpath} | {e}")
    print(f"✔ Duplicate images removed: {removed}\n")


# STEP 4 — Merge into binary classes + resize
def merge_and_resize():
    for split in ['train', 'test']:
        for orig_cls, binary_cls in CLASS_MAP.items():
            src_dir = os.path.join(SOURCE, split, orig_cls)
            dst_dir = os.path.join(TARGET, split, binary_cls)
            if not os.path.isdir(src_dir):
                print(f"  Skipping missing folder: {src_dir}")
                continue
            copied = 0
            for fname in os.listdir(src_dir):
                src_path = os.path.join(src_dir, fname)
                dst_path = os.path.join(dst_dir, f"{orig_cls}_{fname}")
                try:
                    with Image.open(src_path) as img:
                        img.convert('RGB').resize(SIZE, Image.LANCZOS).save(dst_path)
                        copied += 1
                except Exception as e:
                    print(f"  Skipping {src_path}: {e}")
            print(f"  {split}/{orig_cls} → {split}/{binary_cls} | {copied} images")
    print("✔ Merge and resize complete.\n")


# STEP 5 — Check class balance
def check_balance():
    for split in ['train', 'test']:
        print(f"  --- {split.upper()} ---")
        split_path = os.path.join(TARGET, split)
        counts = {}
        for cls in sorted(os.listdir(split_path)):
            cls_path = os.path.join(split_path, cls)
            if os.path.isdir(cls_path):
                n = len(os.listdir(cls_path))
                counts[cls] = n
                print(f"    {cls:10s}: {n} images")
        if counts:
            ratio = max(counts.values()) / min(counts.values())
            print(f"    Imbalance ratio: {ratio:.2f}x")
            if ratio > 2:
                print("    ⚠ Consider using class_weight during training.")
    print("\n✔ Balance check complete.")


# ─── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 45)
    print("   DROWSINESS DATASET CLEANING PIPELINE")
    print("=" * 45 + "\n")

    create_folders()
    remove_corrupted()
    remove_duplicates()
    merge_and_resize()
    check_balance()

    print("\n✅ All done! dataset_binary/ is ready for training.")
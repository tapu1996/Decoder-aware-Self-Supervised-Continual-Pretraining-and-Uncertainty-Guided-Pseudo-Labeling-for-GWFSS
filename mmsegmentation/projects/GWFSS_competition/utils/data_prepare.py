import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

# Set seed for reproducibility
random.seed(42)

# Source directories
source_root = Path("gwfss_competition_train")
image_dir = source_root / "images"
mask_dir = source_root / "class_id"

# Target directories
target_root = Path("GWFSS")
img_train = target_root / "img_dir" / "train"
img_test = target_root / "img_dir" / "test"
ann_train = target_root / "ann_dir" / "train"
ann_test = target_root / "ann_dir" / "test"

# Create folders
for d in [img_train, img_test, ann_train, ann_test]:
    d.mkdir(parents=True, exist_ok=True)

# Group images by domain
domain_groups = defaultdict(list)
for file in sorted(os.listdir(image_dir)):
    if file.endswith(".png"):
        domain = file.split("_")[0]
        domain_groups[domain].append(file)

print(f"\nFound {len(domain_groups)} domains.")
total_train = total_test = 0

# Split and copy
for domain, files in domain_groups.items():
    if len(files) != 11:
        print(f"Domain {domain} has {len(files)} images (expected 11). Skipping.")
        continue

    test_files = random.sample(files, 2)
    train_files = [f for f in files if f not in test_files]

    for f in train_files:
        shutil.copy(image_dir / f, img_train / f)
        shutil.copy(mask_dir / f, ann_train / f)
    for f in test_files:
        shutil.copy(image_dir / f, img_test / f)
        shutil.copy(mask_dir / f, ann_test / f)

    print(f"✔ Domain {domain}: {len(train_files)} train, {len(test_files)} test")
    total_train += len(train_files)
    total_test += len(test_files)

print(f"Split complete. Total: {total_train} train, {total_test} test images.")
print(f"Saved in: {target_root}")

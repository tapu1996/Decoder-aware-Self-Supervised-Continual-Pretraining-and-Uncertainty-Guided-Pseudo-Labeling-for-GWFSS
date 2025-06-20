import os
import random
import shutil
from collections import defaultdict

# Set seed for reproducibility
random.seed(42)

# Original paths
img_dir = "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/data/gwfss_competition_train/images"
ann_dir = "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/data/gwfss_competition_train/class_id"

# Output base directory
output_base = "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/data/folds"

# Load image names
all_images = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])

# Group by domain
domain_dict = defaultdict(list)
for img in all_images:
    domain = img.split("_")[0]  # domain1, domain2, ..., domain9
    domain_dict[domain].append(img)

# Shuffle each domain's list
for domain in domain_dict:
    random.shuffle(domain_dict[domain])

# Split each domain into 5 roughly equal chunks
domain_folds = defaultdict(list)
for domain, imgs in domain_dict.items():
    fold_chunks = [[] for _ in range(5)]
    for i, img in enumerate(imgs):
        fold_chunks[i % 5].append(img)
    domain_folds[domain] = fold_chunks

# Process 5 folds
for fold in range(5):
    fold_name = f"Fold_{fold + 1}"
    print(f"\n Creating {fold_name}...")

    # Target directories
    img_train_dir = os.path.join(output_base, fold_name, "img_dir", "train")
    img_val_dir   = os.path.join(output_base, fold_name, "img_dir", "test")
    ann_train_dir = os.path.join(output_base, fold_name, "ann_dir", "train")
    ann_val_dir   = os.path.join(output_base, fold_name, "ann_dir", "test")

    # Create directories
    for path in [img_train_dir, img_val_dir, ann_train_dir, ann_val_dir]:
        os.makedirs(path, exist_ok=True)

    train_imgs = []
    val_imgs = []

    # For each domain, assign 1 fold's images to val, rest to train
    for domain, folds in domain_folds.items():
        for i in range(5):
            if i == fold:
                val_imgs.extend(folds[i])
            else:
                train_imgs.extend(folds[i])

    # Function to copy images & annotations
    def copy_files(file_list, img_out, ann_out):
        for file in file_list:
            shutil.copy2(os.path.join(img_dir, file), os.path.join(img_out, file))
            shutil.copy2(os.path.join(ann_dir, file), os.path.join(ann_out, file))

    # Copy files
    copy_files(train_imgs, img_train_dir, ann_train_dir)
    copy_files(val_imgs, img_val_dir, ann_val_dir)

    print(f" {fold_name} created: {len(train_imgs)} train, {len(val_imgs)} val")

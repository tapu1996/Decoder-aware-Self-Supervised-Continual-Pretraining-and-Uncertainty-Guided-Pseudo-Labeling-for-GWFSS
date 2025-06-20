import os
import random
import shutil
from collections import defaultdict

# Source directories
base_dir = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/data/synth_datasetAll'
img_dir = os.path.join(base_dir, 'img_dir')
cls_dir = os.path.join(base_dir, 'cls_dir')
ann_dir = os.path.join(base_dir, 'ann_dir')

# Destination directories
selected_base = os.path.join(base_dir, 'selected')
selected_img_dir = os.path.join(selected_base, 'img_dir')
selected_cls_dir = os.path.join(selected_base, 'cls_dir')
selected_ann_dir = os.path.join(selected_base, 'ann_dir')

# Create output folders
os.makedirs(selected_img_dir, exist_ok=True)
os.makedirs(selected_cls_dir, exist_ok=True)
os.makedirs(selected_ann_dir, exist_ok=True)

# Group images by domain
all_images = [f for f in os.listdir(img_dir) if f.endswith('.png')]
domain_dict = defaultdict(list)
for fname in all_images:
    domain = fname.split('_')[0]  # e.g., 'domain1'
    domain_dict[domain].append(fname)

# Randomly select 12 per domain and copy files
for domain, files in domain_dict.items():
    sampled = random.sample(files, min(12, len(files)))
    print(f"[{domain}] Selected: {sampled}")

    for fname in sampled:
        # Image
        shutil.copy(os.path.join(img_dir, fname), os.path.join(selected_img_dir, fname))

        # Class mask
        shutil.copy(os.path.join(cls_dir, fname), os.path.join(selected_cls_dir, fname))

        # Annotation (assuming same name but in ann_dir)
        ann_path = os.path.join(ann_dir, fname)
        if os.path.exists(ann_path):
            shutil.copy(ann_path, os.path.join(selected_ann_dir, fname))
        else:
            print(f"Warning: Annotation not found for {fname}")

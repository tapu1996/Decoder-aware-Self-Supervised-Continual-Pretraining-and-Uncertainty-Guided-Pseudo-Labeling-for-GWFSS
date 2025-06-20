import os
import numpy as np
from collections import Counter
from PIL import Image

# Set this to your mask folder
mask_dir = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/data/gwfss_competition_train/masks'
cat_max_ratio = 0.75  # threshold
valid_exts = ['.png', '.npy']  # depending on format

dominated_images = []

for fname in os.listdir(mask_dir):
    if not any(fname.endswith(ext) for ext in valid_exts):
        continue

    fpath = os.path.join(mask_dir, fname)

    # Load mask
    if fname.endswith('.npy'):
        mask = np.load(fpath)
    else:
        mask = np.array(Image.open(fpath))

    total_pixels = mask.size
    class_counts = Counter(mask.flatten())
    max_class = max(class_counts, key=class_counts.get)
    max_ratio = class_counts[max_class] / total_pixels

    if max_ratio > cat_max_ratio:
        dominated_images.append((fname, max_ratio, dict(class_counts)))

# Report results
print(f"Found {len(dominated_images)} images with >{cat_max_ratio*100:.1f}% of a single class.")

# Print examples
for fname, ratio, counts in dominated_images[:10]:
    print(f"{fname}: {ratio*100:.1f}% single class, class distribution = {counts}")

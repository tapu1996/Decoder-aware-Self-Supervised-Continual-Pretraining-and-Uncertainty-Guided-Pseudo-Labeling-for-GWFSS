import os
import shutil

src_root = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/data/gwfss_augmented'   # ← your current augmented root
dst_root = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/data/gwfss_augmented_flat'

# Target dirs
dst_img_dir = os.path.join(dst_root, 'images')
dst_mask_dir = os.path.join(dst_root, 'class_id')
os.makedirs(dst_img_dir, exist_ok=True)
os.makedirs(dst_mask_dir, exist_ok=True)

# Loop over each domain folder
for domain_name in sorted(os.listdir(src_root)):
    domain_path = os.path.join(src_root, domain_name)
    img_dir = os.path.join(domain_path, 'images')
    mask_dir = os.path.join(domain_path, 'class_id')

    if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
        continue

    for fname in os.listdir(img_dir):
        src = os.path.join(img_dir, fname)
        dst = os.path.join(dst_img_dir, fname)
        shutil.copy(src, dst)

    for fname in os.listdir(mask_dir):
        src = os.path.join(mask_dir, fname)
        dst = os.path.join(dst_mask_dir, fname)
        shutil.copy(src, dst)

print("✅ All domain images and masks flattened to gwfss_augmented_flat/")

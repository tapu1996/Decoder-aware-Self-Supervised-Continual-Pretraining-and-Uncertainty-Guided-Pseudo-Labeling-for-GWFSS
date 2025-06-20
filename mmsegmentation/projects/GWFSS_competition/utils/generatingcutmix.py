import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict

base_dir = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/data/gwfss_competition_train'
out_dir = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/data/gwfss_augmented'

img_dir = os.path.join(base_dir, 'images')
mask_dir = os.path.join(base_dir, 'class_id')

target_class = 2  # Class to boost using ClassMix

# Group files by domain prefix (e.g., domain1)
image_groups = defaultdict(list)
for fname in sorted(os.listdir(img_dir)):
    if fname.endswith(('.jpg', '.png', '.jpeg')) and 'domain' in fname:
        domain = fname.split('_')[0]  # domain1, domain2, ...
        image_groups[domain].append(fname)

def load_pair(f1, f2):
    img1 = cv2.imread(os.path.join(img_dir, f1))
    img2 = cv2.imread(os.path.join(img_dir, f2))
    mask1 = cv2.imread(os.path.join(mask_dir, f1.replace('.jpg', '.png')), 0)
    mask2 = cv2.imread(os.path.join(mask_dir, f2.replace('.jpg', '.png')), 0)

    if img1 is None or img2 is None or mask1 is None or mask2 is None:
        return None, None, None, None

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        mask2 = cv2.resize(mask2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_NEAREST)

    return img1, mask1, img2, mask2

def cutmix(img1, mask1, img2, mask2, beta=1.0):
    lam = np.random.beta(beta, beta)
    h, w, _ = img1.shape
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(w * cut_rat), int(h * cut_rat)
    
    cx = np.random.randint(w)  # ← FIXED
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    new_img = img1.copy()
    new_mask = mask1.copy()
    new_img[bby1:bby2, bbx1:bbx2] = img2[bby1:bby2, bbx1:bbx2]
    new_mask[bby1:bby2, bbx1:bbx2] = mask2[bby1:bby2, bbx1:bbx2]
    return new_img, new_mask


def classmix(img1, mask1, img2, mask2, class_idx):
    new_img = img1.copy()
    new_mask = mask1.copy()
    region = (mask2 == class_idx)
    for c in range(3):
        new_img[:, :, c] = np.where(region, img2[:, :, c], new_img[:, :, c])
    new_mask = np.where(region, class_idx, new_mask)
    return new_img, new_mask

# Loop over each domain
for domain, file_list in image_groups.items():
    if len(file_list) < 2:
        print(f"⚠️ Skipping {domain}: not enough images.")
        continue

    print(f"\n🔧 Augmenting domain: {domain} ({len(file_list)} images)")
    
    # Create output dirs per domain
    domain_img_out = os.path.join(out_dir, domain, 'images')
    domain_mask_out = os.path.join(out_dir, domain, 'class_id')
    os.makedirs(domain_img_out, exist_ok=True)
    os.makedirs(domain_mask_out, exist_ok=True)

    total_aug = len(file_list) * 4  # 2 CutMix + 2 ClassMix per original image
    aug_idx = 0

    for i in tqdm(range(total_aug), desc=f'Augmenting {domain}'):
        f1, f2 = random.sample(file_list, 2)
        img1, mask1, img2, mask2 = load_pair(f1, f2)
        if img1 is None:
            continue

        if i % 4 < 2:
            new_img, new_mask = cutmix(img1, mask1, img2, mask2)
            name = f'{domain}_cutmix_{aug_idx:04}.jpg'
        else:
            new_img, new_mask = classmix(img1, mask1, img2, mask2, target_class)
            name = f'{domain}_classmix_{aug_idx:04}.jpg'

        cv2.imwrite(os.path.join(domain_img_out, name), new_img)
        cv2.imwrite(os.path.join(domain_mask_out, name.replace('.jpg', '.png')), new_mask)
        aug_idx += 1

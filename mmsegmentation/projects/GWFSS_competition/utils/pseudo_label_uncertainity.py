import os
import subprocess
import torch
import numpy as np
from mmengine.fileio import load
from collections import defaultdict
from PIL import Image

import argparse
import glob

dom = 4
#make sure to change in dataloader as well


# --- CONFIGURATION ---
CONFIG = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/configs/ConvNextUpperNet_large_512f0.py'
TEST_SCRIPT = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/tools/test.py'

OUTPUT_DIR = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/data/2StagePseudoLabel/stage2'
PRED_DIR = os.path.join(OUTPUT_DIR, str(dom), 'individual_preds')
PSEUDO_LABEL_DIR = os.path.join(OUTPUT_DIR, str(dom), 'pseudo_labels')
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(PSEUDO_LABEL_DIR, exist_ok=True)

CHECKPOINTS = [
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/ConvNextKfold/fold1/best_mIoU_iter_4400.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/ConvNextKfold/fold2/best_mIoU_iter_8200.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/ConvNextKfold/fold3/best_mIoU_iter_5400.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/ConvNextKfold/fold4/best_mIoU_iter_4600.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/ConvNextKfold/fold5/best_mIoU_iter_15800.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/DeconMLKfold/fold1/best_mIoU_iter_5800.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/DeconMLKfold/fold2/best_mIoU_iter_20000.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/DeconMLKfold/fold3/best_mIoU_iter_5400.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/DeconMLKfold/fold4/best_mIoU_iter_5400.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/DeconMLKfold/fold5/best_mIoU_iter_9400.pth"
]

#--- STEP 1: Run TTA Inference for All Models ---
for i, ckpt in enumerate(CHECKPOINTS):
    out_pkl = os.path.join(PRED_DIR, f'fold{i+1}_results.pkl')

    cmd = [
        'python', TEST_SCRIPT,
        CONFIG,
        ckpt,
        '--tta',
        '--out', out_pkl,
    ]

    print(f"\n🚀 Running fold {i+1} with TTA...")
    subprocess.run(cmd, check=True)

# --- STEP 2: Aggregate Logits from All Models ---
NUM_MODELS = len(CHECKPOINTS)
image_logits_sum = defaultdict(lambda: None)

for i in range(NUM_MODELS):
    pkl_path = os.path.join(PRED_DIR, f'fold{i+1}_results.pkl')
    preds = load(pkl_path)

    for data in preds:
        img_path = data['img_path']
        logits = torch.tensor(data['seg_logits']['data'])  # (C, H, W)

        if image_logits_sum[img_path] is None:
            image_logits_sum[img_path] = logits.clone()
        else:
            image_logits_sum[img_path] += logits

UNCERTAINTY_DIR = os.path.join(OUTPUT_DIR, 'uncertainty_maps')
os.makedirs(UNCERTAINTY_DIR, exist_ok=True)
uncertainty_scores = {}

for img_path, sum_logits in image_logits_sum.items():
    avg_logits = sum_logits / NUM_MODELS
    probs = torch.softmax(avg_logits, dim=0)  # [C, H, W]

    entropy_map = -torch.sum(probs * torch.log(probs + 1e-12), dim=0).cpu().numpy()  # [H, W]
    uncertainty_score = float(np.mean(entropy_map))  # Scalar image-level uncertainty

    pred_mask = torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8)
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # Save pseudo-label
    save_path = os.path.join(PSEUDO_LABEL_DIR, f"{base_name}.png")
    Image.fromarray(pred_mask).save(save_path)

    # Save entropy map
    entropy_path = os.path.join(UNCERTAINTY_DIR, f"{base_name}.npy")
    np.save(entropy_path, entropy_map)

    # Save uncertainty score
    uncertainty_scores[base_name] = uncertainty_score
    print(f"✅ {base_name} | Uncertainty: {uncertainty_score:.4f} | Saved pseudo-label and entropy.")

import pandas as pd

df = pd.DataFrame(list(uncertainty_scores.items()), columns=['Image', 'UncertaintyScore'])
df.to_csv(os.path.join(OUTPUT_DIR, 'uncertainty_scores.csv'), index=False)
print("📁 Saved uncertainty_scores.csv")

# # --- STEP 3: Save Pseudo-Labels as PNGs ---
# for img_path, sum_logits in image_logits_sum.items():
#     avg_logits = sum_logits / NUM_MODELS
#     probs = torch.softmax(avg_logits, dim=0)
#     pred_mask = torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8)

#     # Extract filename (handle .jpg/.png safely)
#     base_name = os.path.splitext(os.path.basename(img_path))[0]
#     save_path = os.path.join(PSEUDO_LABEL_DIR, f"{base_name}.png")

#     Image.fromarray(pred_mask).save(save_path)
#     print(f"✅ Saved pseudo-label: {save_path}")

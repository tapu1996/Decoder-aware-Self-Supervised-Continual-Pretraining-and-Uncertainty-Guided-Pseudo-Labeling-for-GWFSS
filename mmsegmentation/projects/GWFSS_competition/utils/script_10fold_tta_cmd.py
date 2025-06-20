import os
import subprocess
import pickle
import torch
import numpy as np
import pandas as pd
from mmengine.fileio import load

# --- CONFIGURATION ---
CONFIG = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/configs/ConvNextUpperNet_large_512f0.py'
test_images_dir = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/validation_competition/gwfss_competition_val/images'
output_dir = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/validation_competition/prediction_ensemble'

TEST_SCRIPT = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/tools/test.py'  # assumes OpenMMLab 2.x tools/test.py
OUTPUT_DIR = output_dir
PRED_DIR = os.path.join(OUTPUT_DIR, 'individual_preds')
CSV_DIR = os.path.join(OUTPUT_DIR, 'csv_preds')
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

os.makedirs(output_dir, exist_ok=True)

# --- CHECKPOINT FILES ---
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

# --- STEP 1: Run TTA Test for All Models ---
for i, ckpt in enumerate(CHECKPOINTS):
    out_pkl = os.path.join(PRED_DIR, f'fold{i+1}_results.pkl')

    cmd = [
        'python', TEST_SCRIPT,
        CONFIG,
        ckpt,
        '--tta',
        '--out', out_pkl,
    ]

    print(f"\n Running fold {i+1} with TTA...")
    subprocess.run(cmd, check=True)


import os
import torch
import torch.nn.functional as F
import pandas as pd
from mmengine.fileio import load
from collections import defaultdict

# --- CONFIGURATION ---


NUM_MODELS = len(CHECKPOINTS)  # Total .pkl files

# --- STEP 1: Sum Logits Per Image ---
image_logits_sum = defaultdict(lambda: None)

for i in range(NUM_MODELS):
    pkl_path = os.path.join(PRED_DIR, f'fold{i+1}_results.pkl')
    preds = load(pkl_path)

    for data in preds:
        img_path = data['img_path']
        logits = data['seg_logits']['data']  # Tensor (C, H, W)

        if image_logits_sum[img_path] is None:
            image_logits_sum[img_path] = logits.clone()
        else:
            image_logits_sum[img_path] += logits

# --- STEP 2: Average, Predict, Save ---
for img_path, sum_logits in image_logits_sum.items():
    avg_logits = sum_logits / NUM_MODELS
    probs = F.softmax(avg_logits, dim=0)
    pred_mask = torch.argmax(probs, dim=0).cpu().numpy()

    # Save as CSV
    image_name = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join(CSV_DIR, f"{image_name}.csv")
    pd.DataFrame(pred_mask.astype(np.uint8)).to_csv(output_path, index=False, header=None)

    print(f"✅ Saved: {output_path}")


# # --- STEP 2: Load All Predictions ---
# print("\n📥 Loading predictions...")
# all_preds = []
# for i in range(len(CHECKPOINTS)):
#     pkl_path = os.path.join(PRED_DIR, f'fold{i+1}_results.pkl')
#     preds = load(pkl_path)  # list of dicts, each with 'pred_sem_seg': (C, H, W)
#     all_preds.append(preds)
#     print(all_preds)

# # --- STEP 3: Average & Save Final Predictions ---
# num_images = len(all_preds[0])
# print(f"\n📊 Averaging {len(CHECKPOINTS)} models over {num_images} images...")

# for idx in range(num_images):
#     # stack predictions: [(C, H, W), ..., (C, H, W)]
#     logits_list = [torch.tensor(fold_preds[idx]['seg_logits']['data']).float()  for fold_preds in all_preds]
#     avg_logits = torch.stack(logits_list).mean(dim=0)  # (C, H, W)

#     # convert to class prediction
#     probs = torch.softmax(avg_logits, dim=0)
#     pred_mask = torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8)  # (H, W)

#     # save as CSV
#     out_csv_path = os.path.join(CSV_DIR, f'sample_{idx:04d}.csv')
#     pd.DataFrame(pred_mask).to_csv(out_csv_path, index=False, header=None)
#     print(f"✅ Saved: {out_csv_path}")
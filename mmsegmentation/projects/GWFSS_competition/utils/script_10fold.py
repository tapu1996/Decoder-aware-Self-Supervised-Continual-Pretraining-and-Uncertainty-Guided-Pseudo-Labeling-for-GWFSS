import os
import mmcv
import pandas as pd
import numpy as np
from mmseg.apis import init_model, inference_model
from mmengine.config import Config
import torch.nn.functional as F
import torch

# --- CONFIGURATION ---
config_file = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/configs/ConvNextUpperNet_large_512f0.py'
test_images_dir = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/validation_competition/gwfss_competition_val/images'
output_dir = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/validation_competition/prediction_ensemble'
os.makedirs(output_dir, exist_ok=True)

# --- CHECKPOINT FILES ---
checkpoint_files = [
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

# --- LOAD MODELS ---
models = []
for ckpt in checkpoint_files:
    model = init_model(config_file, ckpt, device='cuda:0')
    models.append(model)

# --- INFERENCE ---
image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


for image_file in image_files:
    image_path = os.path.join(test_images_dir, image_file)
    print(f"Processing: {image_path}")

    sum_logits = None

    for model in models:
        result = inference_model(model, image_path)
        logits = result.seg_logits.data  # Shape: (num_classes, H, W)
        if sum_logits is None:
            sum_logits = logits.clone()
        else:
            sum_logits += logits

    avg_logits = sum_logits / len(models)
    probs = F.softmax(avg_logits, dim=0)
    pred_mask = torch.argmax(probs, dim=0).cpu().numpy()


    # Save as CSV
    image_name = os.path.splitext(image_file)[0]
    output_path = os.path.join(output_dir, f"{image_name}.csv")
    pd.DataFrame(pred_mask.astype(np.uint8)).to_csv(output_path, index=False, header=None)

    print(f"Saved ensemble prediction: {output_path}")

import os
import mmcv
import pandas as pd
import numpy as np
from mmseg.apis import init_model, inference_model
from mmengine.config import Config
import torch.nn.functional as F
import torch
# --- CONFIGURATION ---
config_file = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/configs/ConvNextUpperNet_large_512.py'  # Replace with your config file
checkpoint_file = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/ConvNext-Large_im1k_highLRfinetuning_woCatmix/best_mIoU_iter_6900.pth'
test_images_dir = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/validation_competition/gwfss_competition_val/images'  # Folder containing test images (e.g., .png)
output_dir = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/validation_competition/prediction'  # Where to save .csv predictions


os.makedirs(output_dir, exist_ok=True)

# --- INITIALIZE MODEL ---
model = init_model(config_file, checkpoint_file, device='cuda:0')  # or 'cpu'
#model.cfg.test_dataloader.dataset.pipeline = model.cfg.tta_pipeline

# --- INFERENCE ---
image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#print(image_file)
for image_file in image_files:
    image_path = os.path.join(test_images_dir, image_file)
    print(image_path)
    result = inference_model(model, image_path)

    # Get predicted mask
    predicted_mask = result.pred_sem_seg.data[0].cpu().numpy()  # (H, W) with class labels

    # Save as CSV
    image_name = os.path.splitext(image_file)[0]  # Remove extension
    output_path = os.path.join(output_dir, f"{image_name}.csv")
    pd.DataFrame(predicted_mask.astype(np.uint8)).to_csv(output_path, index=False, header=None)

    print(f"Saved prediction: {output_path}")

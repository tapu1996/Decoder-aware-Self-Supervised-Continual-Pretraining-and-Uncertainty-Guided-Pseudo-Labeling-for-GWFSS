import os
import mmcv
import pandas as pd
import numpy as np
from mmseg.apis import init_model, inference_model
from mmengine.config import Config
import torch.nn.functional as F
import torch
from PIL import Image
import shutil

# --- CONFIGURATION ---
config_file_beit = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/configs/beit_large_512f0.py'
config_file = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/configs/ConvNextUpperNet_large.py'
test_images_dir_root = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/data/2StagePseudoLabel/stage1'
output_dir = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/data/2StagePseudoLabel/stage2_100'

top11_dir = os.path.join(output_dir, 'top11')
top11_img_dir = os.path.join(top11_dir, 'images')
top11_mask_dir = os.path.join(top11_dir, 'masks')
top11_uncertainty_dir = os.path.join(top11_dir, 'uncertainity')


os.makedirs(output_dir, exist_ok=True)

os.makedirs(top11_img_dir, exist_ok=True)
os.makedirs(top11_mask_dir, exist_ok=True)
os.makedirs(top11_uncertainty_dir, exist_ok=True)

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


beit_checkpoint_files = [
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/beitFolds/beitfold0/best_mIoU_iter_13200.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/beitFolds/beitfold1/best_mIoU_iter_12400.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/beitFolds/beitfold2/best_mIoU_iter_4800.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/beitFolds/beitfold3/best_mIoU_iter_15600.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/beitFolds/beitfold4/best_mIoU_iter_13200.pth"


]


# --- LOAD MODELS ---
models = []
for ckpt in checkpoint_files:
    model = init_model(config_file, ckpt, device='cuda:0')
    models.append(model)

for ckpt in beit_checkpoint_files:
    model = init_model(config_file_beit, ckpt, device='cuda:0')
    models.append(model)



for tt in range(1,10):
    pseudo_label_dir = os.path.join(output_dir,str(tt), 'pseudo_labels')
    uncertainty_dir = os.path.join(output_dir,str(tt), 'uncertainty_maps')
    os.makedirs(pseudo_label_dir, exist_ok=True)
    os.makedirs(uncertainty_dir, exist_ok=True)
    test_images_dir = os.path.join(test_images_dir_root,str(tt))
    # --- INFERENCE ---
    image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    uncertainty_scores = []
    all_outputs = {}


    for image_file in image_files:
        image_path = os.path.join(test_images_dir, image_file)
        image_name = os.path.splitext(image_file)[0]
        print(f"Processing: {image_path}, {image_name}")

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
        entropy_map = -torch.sum(probs * torch.log(probs + 1e-8), dim=0) 
        pred_mask = torch.argmax(probs, dim=0).cpu().numpy()
        entropy_map_np = entropy_map.cpu().numpy()
        norm_entropy = (entropy_map_np - entropy_map_np.min()) / (entropy_map_np.max() - entropy_map_np.min() + 1e-8)
        entropy_uint8 = (norm_entropy * 255).astype(np.uint8)
        mask_png_path = os.path.join(pseudo_label_dir, f"{image_name}.png")
        entropy_png_path = os.path.join(uncertainty_dir, f"{image_name}.png")
        Image.fromarray(pred_mask.astype(np.uint8)).save(mask_png_path)
        Image.fromarray(entropy_uint8).save(entropy_png_path)
        avg_entropy = float(entropy_map_np.mean())
        uncertainty_scores.append((image_name, avg_entropy))
        all_outputs[image_name] = {
            "mask": pred_mask,
            "entropy": entropy_uint8,
            "image_path": image_path
            }

        print(f"Saved PNG mask & entropy for {image_name} | Uncertainty: {avg_entropy:.4f}")
    
    uncertainty_scores.sort(key=lambda x: x[1])
    top11 = uncertainty_scores[:11]
    for image_name, score in top11:
        mask = all_outputs[image_name]["mask"]
        entropy = all_outputs[image_name]["entropy"]
        original_img_path = all_outputs[image_name]["image_path"]

        mask_path = os.path.join(top11_mask_dir, f"{image_name}.png")
        entropy_path = os.path.join(top11_uncertainty_dir, f"{image_name}.png")
        img_path = os.path.join(top11_img_dir, f"{image_name}.png")

        Image.fromarray(mask.astype(np.uint8)).save(mask_path)
        Image.fromarray(entropy).save(entropy_path)
        shutil.copy2(original_img_path, img_path)

        print(f"📌 Top-11 saved: {image_name} | Uncertainty: {score:.4f}")
    
    df = pd.DataFrame(uncertainty_scores, columns=['Image', 'UncertaintyScore'])
    df.to_csv(os.path.join(output_dir, f'uncertainty_scores-{tt}.csv'), index=False)
    print("Saved uncertainty_scores.csv")



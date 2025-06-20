import os
import mmcv
import pandas as pd
import numpy as np
from mmseg.apis import init_model, inference_model
from mmengine.config import Config
import torch.nn.functional as F
import torch

# --- CONFIGURATION ---

#change the path of the config files#

#Replace all the paths "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation" with your mmsegmentation path

config_file = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/configs/ConvNextUpperNet_large_768.py'
config_file_beit = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/configs/beit_large_512f0.py'
#refer to image dir
test_images_dir = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/data/test'
#refer to output_dir
output_dir = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/validation_competition/test'
os.makedirs(output_dir, exist_ok=True)


#Beit checkpoint files
#Replace all the paths "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs" with your checkpoint root dir
beit_checkpoint_files = [
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/beitFolds/beitfold0/best_mIoU_iter_13200.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/beitFolds/beitfold1/best_mIoU_iter_12400.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/beitFolds/beitfold2/best_mIoU_iter_4800.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/beitFolds/beitfold3/best_mIoU_iter_15600.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/beitFolds/beitfold4/best_mIoU_iter_13200.pth"

]

#All convnext checkpoint files

checkpoint_files=[
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/DeconPseudoFirstMainSecond/fold0/best_mIoU_iter_3800.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/DeconPseudoFirstMainSecond/fold1/best_mIoU_iter_4400.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/DeconPseudoFirstMainSecond/fold2/best_mIoU_iter_4200.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/DeconPseudoFirstMainSecond/fold3/best_mIoU_iter_2600.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/DeconPseudoFirstMainSecond/fold4/best_mIoU_iter_7200.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/ConvnextPseudoFirstMainSecond/fold0/best_mIoU_iter_3200.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/ConvnextPseudoFirstMainSecond/fold1/best_mIoU_iter_12000.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/ConvnextPseudoFirstMainSecond/fold2/best_mIoU_iter_4000.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/ConvnextPseudoFirstMainSecond/fold3/best_mIoU_iter_5600.pth",
    "/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/work_dirs/ConvnextPseudoFirstMainSecond/fold4/best_mIoU_iter_6200.pth"
    ]

# --- LOAD MODELS ---
models = []
for ckpt in beit_checkpoint_files:
    model = init_model(config_file_beit, ckpt, device='cuda:0')
    models.append(model)

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

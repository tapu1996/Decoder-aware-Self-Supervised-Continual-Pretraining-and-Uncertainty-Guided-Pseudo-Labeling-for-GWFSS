import torch
import torch.nn.functional as F
from mmseg.utils import register_all_modules
from mmengine.registry import MODELS
from mmengine.runner import load_checkpoint
import torch
import torch.nn.functional as F
from mmseg.utils import register_all_modules
from mmengine.registry import MODELS
from mmengine.runner import load_checkpoint
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from PIL import Image
import os
from tqdm import tqdm
import shutil
import argparse
import glob
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="Top-K Unlabeled Image Selection Based on Cosine Similarity")

parser.add_argument('--domain', type=int, required=True,
                    help='Path to labeled image directory')

dom = str(args.domain)

# ----Change Paths ----
labeled_dir = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/data/gwfss_competition_train/images'
unlabeled_dir = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/data/gwfss_competition_pretrain'
ckpt_path = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/model/deconMLCOntinual/deconmlCOntinual_enc.pth'
dst_dir = f"/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/data/2StagePseudoLabel/stage1/{dom}"
top_k = 100
# ----Change Paths ----

args = parser.parse_args()
# Register everything in MMSeg
register_all_modules()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Your config
convnext_cfg = dict(
    type='ConvNeXt',
    arch='large',
    out_indices=(3,),
    drop_path_rate=0.4,
    layer_scale_init_value=1.0,
    gap_before_final_norm=True,
    init_cfg=dict(
        type='Pretrained',
        checkpoint=ckpt_path
    )
)

# Build model
#model = MODELS.build(convnext_cfg)

convnext_cfg.pop('init_cfg', None)
model = MODELS.build(convnext_cfg)

# Now manually load checkpoint
load_checkpoint(model, ckpt_path, map_location='cpu', revise_keys=[(r'^backbone\.', '')])
model.to(device).eval()


# --------- 4. Preprocessing ---------
transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --------- 5. Image path ---------
# img_path = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/data/GWFSS/img_dir/train/domain1_00000.png'  # <<< CHANGE THIS

# # --------- 6. Extract features ---------
# img = default_loader(img_path)  # loads and converts to RGB
# img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    feat_map = model(img_tensor)[0]  # [1, 1536, 7, 7] for ConvNeXt-Large
    #pooled_feat = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)  # [1, 1536]
    print("Feature vector shape:", feat_map.shape)
    #print("Feature vector mean:", pooled_feat.mean().item())

def extract_feature(image_path):
    img = default_loader(image_path)
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat_map = model(img_tensor)[0]  # shape: [1, C]
        #pooled_feat = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)  # shape: [1, C]
    return feat_map.squeeze(0).cpu()  # shape: [C]



# Recursively collect all image paths in subfolders
all_unlabeled_paths = glob.glob(os.path.join(unlabeled_dir, '**', '*.png'), recursive=True) + \
                      glob.glob(os.path.join(unlabeled_dir, '**', '*.jpg'), recursive=True) + \
                      glob.glob(os.path.join(unlabeled_dir, '**', '*.jpeg'), recursive=True)

fname_to_path = {os.path.basename(p): p for p in all_unlabeled_paths}

# ---- Step 1: Extract labeled features ----
labeled_feats = []
labeled_names = []



for fname in sorted(os.listdir(labeled_dir)):
    if fname.endswith(('.png', '.jpg', '.jpeg')) and fname.startswith(f'domain{dom}_'):
        path = os.path.join(labeled_dir, fname)
        print(path)
        feat = extract_feature(path)
        labeled_feats.append(feat)
        labeled_names.append(fname)

labeled_feats = torch.stack(labeled_feats)  # shape: [10, C]

# ---- Step 2: Extract unlabeled features ----
unlabeled_feats = []
unlabeled_names = []

for path in tqdm(all_unlabeled_paths, desc="Extracting Unlabeled Features"):
    fname = os.path.basename(path)
    feat = extract_feature(path)
    unlabeled_feats.append(feat)
    unlabeled_names.append(fname)

unlabeled_feats = torch.stack(unlabeled_feats)  # shape: [N, C]

# ---- Step 3: Cosine Similarity ----
labeled_feats = F.normalize(labeled_feats, dim=1)  # [10, C]
unlabeled_feats = F.normalize(unlabeled_feats, dim=1)  # [N, C]

similarity_matrix = torch.matmul(labeled_feats, unlabeled_feats.T)  # [10, N]
max_sim = similarity_matrix.mean(dim=0)  # [N]

# ---- Step 4: Top-K most similar ----
topk_sim, topk_indices = torch.topk(max_sim, top_k)
topk_files = [unlabeled_names[i] for i in topk_indices.tolist()]

# ---- Step 5: Print or Save ----
print(f"\nTop-{top_k} closest unlabeled images to labeled set:")
for i, fname in enumerate(topk_files):
    print(f"{i+1:3d}: {fname} (similarity={topk_sim[i]:.4f})")

os.makedirs(dst_dir, exist_ok=True)

for fname in topk_files:
    src_path = fname_to_path.get(fname)
    #src_path = os.path.join(unlabeled_dir, fname)
    dst_path = os.path.join(dst_dir, fname)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
    else:
        print(f"File not found: {src_path}")
print(f"Copied Top-{top_k} images to {dst_dir}")


import pandas as pd

# Create DataFrame from filenames and similarity scores
df = pd.DataFrame({
    'rank': list(range(1, len(topk_files) + 1)),
    'filename': topk_files,
    'similarity': [float(s) for s in topk_sim]
})

# Save to CSV
csv_path = dst_dir + "/" + 'topk_similar_images.csv'
df.to_csv(csv_path, index=False)
print(f" Saved Top-K similarity results to: {csv_path}")




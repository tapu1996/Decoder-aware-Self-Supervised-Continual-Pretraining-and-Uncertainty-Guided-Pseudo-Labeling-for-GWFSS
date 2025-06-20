import os
import torch
import numpy as np
np.float = float
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from torchvision import transforms
from mmseg.utils import register_all_modules
from mmengine.registry import MODELS
from mmengine.runner import load_checkpoint

# --- Setup ---
register_all_modules()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Paths ---
image_dir = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/data/gwfss_competition_train/images'
mask_dir = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/data/gwfss_competition_train/class_id'
ckpt_path = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/model/convnext_imagetnet_large_1k.pth'

# --- Load ConvNeXt backbone ---
convnext_cfg = dict(
    type='ConvNeXt',
    arch='large',
    out_indices=(2,),  # Use last stage
    drop_path_rate=0.4,
    layer_scale_init_value=1.0,
    # gap_before_final_norm=True
)
model = MODELS.build(convnext_cfg)
load_checkpoint(model, ckpt_path, map_location='cpu', revise_keys=[(r'^backbone\.', '')])
model.to(device).eval()

# --- Image transform ---
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Feature Storage ---
class_features = {0: [], 1: [], 2: [], 3: []}  # BG, Stem, Leaf, Spike
class_names = {0: 'Background', 1: 'Stem', 2: 'Leaf', 3: 'Spike'}

# --- Feature extraction ---
def extract_pixel_features(img_path, mask_path):
    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).resize((512, 512), resample=Image.NEAREST)
    mask_np = np.array(mask)

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat_map = model(img_tensor)[0]  # Expected [1, C, H, W]
        feat_map = feat_map.squeeze(0).cpu().numpy()  # Should be [C, H, W]

    if feat_map.ndim != 3:
        raise ValueError(f"Expected 3D feature map after squeeze, got shape: {feat_map.shape}")

    C, H, W = feat_map.shape

    for cls_id in range(4):
        coords = np.argwhere(mask_np == cls_id)
        for y, x in coords[::10]:  # sample every 10th pixel
            if 0 <= y < H and 0 <= x < W:
                vec = feat_map[:, y, x]
                class_features[cls_id].append(vec)

# --- Iterate over all image-mask pairs ---
for fname in tqdm(sorted(os.listdir(image_dir))):
    if fname.startswith('domain') and fname.endswith('.png'):
        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname)
        extract_pixel_features(img_path, mask_path)

# --- PCA + Plot ---
colors = ['gray', 'green', 'blue', 'red']
plt.figure(figsize=(10, 6))

for cls_id, feats in class_features.items():
    if len(feats) > 0:
        feats = np.array(feats)
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(feats)
        plt.scatter(reduced[:, 0], reduced[:, 1], s=5, alpha=0.5,
                    label=class_names[cls_id], color=colors[cls_id])

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Pixel-wise PCA of ConvNeXt Features by Class")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pixelwise_pca_by_class_imagenet.png", dpi=300)
plt.show()
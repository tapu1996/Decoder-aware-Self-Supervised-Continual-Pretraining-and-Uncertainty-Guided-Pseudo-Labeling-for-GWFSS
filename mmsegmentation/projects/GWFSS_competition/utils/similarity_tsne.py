import numpy as np
if not hasattr(np, 'float'):
    np.float = float
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

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        checkpoint='/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/model/deconMLCOntinual/deconmlCOntinual_enc.pth'
    )
)

# Build model
#model = MODELS.build(convnext_cfg)



transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_feature(image_path,model):
    img = default_loader(image_path)
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat_map = model(img_tensor)[0]  # shape: [1, C, H, W]
        #pooled_feat = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)  # shape: [1, C]
    return feat_map.squeeze(0).cpu()  # shape: [C]

checkpoints = [
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

# Now manually load checkpoint
img_dir = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/data/gwfss_competition_val/images'
img_list = os.listdir(img_dir)
img_list = img_list[0:50]

labeled_feats = []
all_tags = []
for path in checkpoints:
    convnext_cfg.pop('init_cfg', None)
    model = MODELS.build(convnext_cfg)
    ckpt_path = path
    load_checkpoint(model, ckpt_path, map_location='cpu', revise_keys=[(r'^backbone\.', '')])
    model.to(device).eval()
    
    for fname in img_list:
        if fname.endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(img_dir, fname)
            feat = extract_feature(path,model)
            labeled_feats.append(feat)
            checkpoints_name_ll =  ckpt_path.split("/")
            ckpt = checkpoints_name_ll[-3] +"_" + checkpoints_name_ll[-2] +"_" + checkpoints_name_ll[-1]
            print(ckpt)
            all_tags.append(ckpt)


features = torch.stack(labeled_feats).numpy()  # [num_ckpts * num_imgs, feat_dim]



# Step 5: Run t-SNE
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
features_tsne = tsne.fit_transform(features)

# Step 6: Plot
plt.figure(figsize=(10, 7))
num_imgs = len(img_list)
colors = plt.cm.get_cmap("tab10", len(checkpoints))

for i, ckpt in enumerate(checkpoints):
    checkpoints_name_ll =  ckpt.split("/")
    labell = checkpoints_name_ll[-3] +"_" + checkpoints_name_ll[-2] 
    idx = list(range(i * num_imgs, (i + 1) * num_imgs))
    plt.scatter(features_tsne[idx, 0], features_tsne[idx, 1], label=labell, alpha=0.7, color=colors(i))

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("t-SNE of Image Representations Across Checkpoints")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()

# Save the plot before showing
plt.savefig("tsne_checkpoints.png", dpi=300, bbox_inches='tight')  # High-resolution output

#plt.show()

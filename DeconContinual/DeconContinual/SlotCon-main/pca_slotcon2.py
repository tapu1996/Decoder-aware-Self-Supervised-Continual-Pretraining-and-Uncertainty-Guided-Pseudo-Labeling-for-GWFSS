import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
from models import resnet
import os
from tqdm import tqdm

from data.datasets import ImageFolder
from models import resnet
from models.slotcon import SlotCon

from pycocotools.coco import COCO
from PIL import Image
import os
from torch.utils.data import Dataset

from pycocotools.coco import COCO
from collections import Counter
from PIL import Image
import os
from torch.utils.data import Dataset

class CocoWithTopCategories(Dataset):
    def __init__(self, img_dir, ann_file, transform=None, top_k=5):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.transform = transform

        # Count category frequency over all annotations
        ann_ids = self.coco.getAnnIds()
        #print(ann_ids)
        anns = self.coco.loadAnns(ann_ids)
        cat_counts = Counter([ann['category_id'] for ann in anns])
        print(cat_counts)
        #exit()
        self.top_cat_ids = list([cat for cat, _ in cat_counts.most_common(top_k)])
        self.top_cat_ids = self.top_cat_ids[1:]
        print(self.top_cat_ids)

        # Filter image IDs: only those with at least one top-k category
        self.ids = []
        for img_id in self.coco.getImgIds():
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            if any(ann['category_id'] in self.top_cat_ids for ann in anns):
                self.ids.append(img_id)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Pick first category from top-k ones
        cat_id = next((ann['category_id'] for ann in anns if ann['category_id'] in self.top_cat_ids), -1)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, cat_id

    def __len__(self):
        return len(self.ids)

# class CocoWithCategory(Dataset):
#     def __init__(self, img_dir, ann_file, transform=None):
#         self.coco = COCO(ann_file)
#         self.img_dir = img_dir
#         self.ids = self.coco.getImgIds()
#         self.transform = transform

#     def __getitem__(self, index):
#         img_id = self.ids[index]
#         ann_ids = self.coco.getAnnIds(imgIds=img_id)
#         anns = self.coco.loadAnns(ann_ids)
#         cat_id = anns[0]['category_id'] if anns else -1  # use first category if multiple

#         path = self.coco.loadImgs(img_id)[0]['file_name']
#         image = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image, cat_id

#     def __len__(self):
#         return len(self.ids)


def denorm(img):
    mean, val = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    img = (img * val[:, None, None] + mean[:, None, None]) * torch.tensor([255, 255, 255])[:, None, None]
    return img.permute(1, 2, 0).cpu().type(torch.uint8)

def get_model(args):
    encoder = resnet.__dict__[args.arch]
    model = SlotCon(encoder, args, use_decoder=args.use_decoder).cuda()
    model.encoder_k = encoder(head_type="multi_layer")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    weights = {k.replace('module.', ''):v for k, v in checkpoint['model'].items()}

    model.load_state_dict(weights, strict=True)

    # print("Missing keys:")
    #print(load_result.missing_keys)

    # print("\nUnexpected keys:")
    #print(load_result.unexpected_keys)
    # same = torch.allclose(pre_weights, post_weights)
    # print("Weights updated:", not same)
    
    model = model.eval()
    return model

if __name__=='__main__':
    model_names = sorted(name for name in resnet.__all__ if name.islower() and callable(resnet.__dict__[name]))

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='COCOval', choices=['COCO', 'COCOplus', 'ImageNet','COCOval','refuge'], help='dataset type')
    parser.add_argument('--data-dir', type=str, default='./data', help='dataset director')
    parser.add_argument('--image-size', type=int, default=224, help='image crop size')
    parser.add_argument('--min-scale', type=float, default=0.08, help='minimum crop scale')
    parser.add_argument('--model_path', type=str, default='output/slotcon_coco_r50_800ep/ckpt_epoch_800.pth')

    # model
    parser.add_argument('--arch', type=str, default='resnet50', choices=# In the provided code
    # snippet, `model_names` is
    # used as a parameter in the
    # argument parser for the
    # `--arch` option. It is not
    # shown in the code snippet
    # provided, but typically
    # `model_names` would be a
    # list or a dictionary
    # containing the available
    # model architectures that can
    # be used as the backbone
    # architecture for the SlotCon
    # model.
    model_names, help='backbone architecture')
    parser.add_argument('--dim-hidden', type=int, default=4096, help='hidden dimension')
    parser.add_argument('--dim-out', type=int, default=256, help='output feature dimension')
    parser.add_argument('--num-prototypes', type=int, default=256, help='number of prototypes')
    parser.add_argument('--num-prototypes-dec', type=int, default=0, help='number of prototypes')
    parser.add_argument('--teacher-momentum', default=0.99, type=float, help='momentum value for the teacher model')
    parser.add_argument('--teacher-temp', default=0.07, type=float, help='teacher temperature')
    parser.add_argument('--student-temp', default=0.1, type=float, help='student temperature')
    parser.add_argument('--center-momentum', default=0.9, type=float, help='momentum for the center')
    parser.add_argument('--group-loss-weight', default=0.5, type=float, help='balancing weight of the grouping loss')
    parser.add_argument('--use-decoder', action='store_true', help='use decoder or not')
    parser.add_argument('--group-loss-weight-dec', default=0.5, type=float, help='balancing weight of the grouping loss for decoder')
    parser.add_argument('--encoder-loss-weight', default=0.5, type=float, help='balancing weight of the encoder loss when there is a decoder')
    parser.add_argument(
        '--decoder-downstream-dataset', 
        type=str, 
        default="cityscapes", 
        help='dataset used in downstream task, it influences the config of the decoder'
    )
    parser.add_argument(
        '--decoder-type', 
        type=str, 
        default="FCN", 
        help='config for decoder for now only supports FCN|FPN.'
    )
    parser.add_argument(
        '-dds',
        '--decoder-deep-supervision', 
        action='store_true',
        help='Use a loss at each level of the decoder.'
    )
    parser.add_argument(    
        "-skdp",
        '--sk-dropout-prob', 
        type=float, 
        default=0.0, 
        help='Probability of not using the lateral features for each level of the decoder'
    )
    parser.add_argument(
        "-skcdp",
        '--sk-channel-dropout-prob', 
        type=float, 
        default=0.0, 
        help='Probability of droping channels in the lateral features for each level of the decoder'
    )

    # optim.
    parser.add_argument('--batch-size', type=int, default=512, help='total batch size')
    parser.add_argument('--base-lr', type=float, default=1.0,
                        help='base learning when batch size = 256. final lr is determined by linear scale')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'lars'], default='sgd', help='optimizer choice')
    parser.add_argument('--warmup-epoch', type=int, default=5, help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--fp16', action='store_true', default=True, help='whether or not to turn on automatic mixed precision')
    parser.add_argument('--start-epoch', type=int, default=1, help='used for resume')
    parser.add_argument('--epochs', type=int, default=800, help='number of training epochs')
    parser.add_argument('-ps', '--previous-scheduler', action='store_true', 
                        help="Do you want to use the old scheduler. Old one was updated to avoid warnings.")
    
    # misc
    parser.add_argument('--output-dir', type=str, default='./output', help='output director')
    parser.add_argument('--auto-resume', action='store_true', help='auto resume from current.pth')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to the latest checkpoint')
    parser.add_argument('--print-freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save-freq', type=int, default=50, help='save frequency')
    parser.add_argument('--seed', type=int, help='Random seed.')
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers per GPU to use')

    # ddp
    parser.add_argument('--no-ddp', action='store_true', help='are you using DDP?')
    parser.add_argument('--cc', action='store_true', help='are you running on Compute Canada?')
    parser.add_argument('--dec_feature', action='store_true', help='are you calculating dec_feature')


    args = parser.parse_args()
    if not args.no_ddp:
        if args.cc:
            local_rank, rank, world_size = _init_distributed_mode_computecan()
            args.local_rank = local_rank
            args.rank = rank
            args.world_size = world_size
        else:
            if os.getenv("LOCAL_RANK", None)  is not None:
                args.local_rank = int(os.environ["LOCAL_RANK"])
            else:
                print("local rank is not defined, you better not be trying to use DDP!")
                args.local_rank = 0
    else:
        # Only one gpu or no GPU
        args.world_size = 1

    mean_vals, std_vals = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=mean_vals, std=std_vals)])

    dataset = ImageFolder(args.dataset, args.data_dir, transform)
    #ann_file = os.path.join(args.data_dir, 'annotations/instances_val2017.json')
    #img_dir = os.path.join(args.data_dir, 'val2017')
    #dataset = CocoWithTopCategories(img_dir, ann_file, transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    args.num_instances = len(dataloader.dataset)

    model = get_model(args).cuda()
    pooled_features = []
    labels =[]
    with torch.no_grad():
        for data in tqdm(dataloader, desc='Feature extracting', leave=False, disable=False):
            feature = model.encoder_k(data.cuda(non_blocking=True))#.mean(dim=(-2, -1))
            if args.dec_feature==True:
                feature = model.decoder_k(feature)
            #print(f1.shape,f2.shape,f3.shape,f4.shape)
            # print(feature["p5"].shape)
            feature_to_work_on = feature[3]#feature["p2"]
            #print(feature_to_work_on.shape)
            pooled = torch.flatten(feature_to_work_on,start_dim=1)
            #print(pooled.shape)
#             attn_weights = torch.mean(feature_to_work_on, dim=1, keepdim=True)  # [B, 1, H, W]

# # Step 2: Apply softmax across spatial locations (flatten then reshape)
#             B, _, H, W = attn_weights.shape
#             attn_weights = attn_weights.view(B, 1, -1)  # [B, 1, H*W]
#             attn_weights = torch.softmax(attn_weights, dim=-1)  # spatial softmax
#             attn_weights = attn_weights.view(B, 1, H, W)  # [B, 1, H, W]

#             # Step 3: Apply attention weights to original feature map
#             weighted_feature = feature_to_work_on * attn_weights  # [B, C, H, W]

            # Step 4: Sum over spatial dimensions (H, W)
            #pooled = weighted_feature.sum(dim=[2, 3])  # [B, C]
            #pooled = F.adaptive_avg_pool2d(feature_to_work_on, 1).squeeze(-1).squeeze(-1)  # [B, 512]
            #print("after pool:", pooled.shape)
            #exit()
            pooled_features.append(pooled.cpu())
            #labels.extend(label.tolist())
            del data, feature, feature_to_work_on
            torch.cuda.empty_cache()
        
    X = torch.cat(pooled_features, dim=0).numpy()  # [N, C]
    #labels = np.array(labels)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
X_pca = PCA(n_components=2).fit(X)
# Explained variance ratio for each principal component
explained_variance = X_pca.explained_variance_ratio_

print("Explained variance ratio:", explained_variance)
print("Total explained variance:", explained_variance.sum())

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X)  # X is your [N, C] feature matrix

centroid = np.mean(X_tsne, axis=0)
distances = np.linalg.norm(X_tsne - centroid, axis=1)
compactness = np.mean(distances)
print("Global Compactness (avg. distance to centroid):", compactness)

plt.figure(figsize=(8, 6))
#scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab20', s=10, alpha=0.7)
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=10, alpha=0.7)

plt.title('fcn-SNE-layer3: Colored by COCO Category ID+ '+str(compactness))
plt.colorbar(scatter)
plt.savefig('fcn.png')    

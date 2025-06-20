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
import torchvision.transforms as T
from pycocotools.coco import COCO
from PIL import Image
import os
from torch.utils.data import Dataset

from pycocotools.coco import COCO
from collections import Counter
from PIL import Image
import os
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import os


def load_image(img_path, image_size=224):
    image = Image.open(img_path).convert('RGB')
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0)
    return image.resize((image_size, image_size)), tensor


def cosine_attention_map(feature_map, target_pixel):
    C, H, W = feature_map.shape
    x = min(W - 1, max(0, target_pixel[0] * W // 224))
    y = min(H - 1, max(0, target_pixel[1] * H // 224))
    idx = y * W + x

    feat = feature_map.view(C, -1)  # [C, H*W]
    target_feat = feat[:, idx].unsqueeze(1)  # [C, 1]
    feat_norm = F.normalize(feat, dim=0)
    target_norm = F.normalize(target_feat, dim=0)
    sim = torch.mm(target_norm.t(), feat_norm).view(H, W)
    return sim.cpu().numpy()


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
    model = model.eval()
    return model

def visualize_attention(img_pil, attention_map, target_pixel, save_path,dec=False):
    blended_final=[]
    for i in range(4):
        attn_map = attention_map[i]
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
        attn_resized = TF.resize(Image.fromarray((attn_map * 255).astype(np.uint8)), img_pil.size)
        heatmap = cm.jet(np.array(attn_resized) / 255.0)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        blended = Image.blend(img_pil.convert('RGB'), Image.fromarray(heatmap), alpha=0.5)
        blended_final.append(blended)

    # Plot side by side
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    axs[0].imshow(img_pil)
    axs[0].scatter([target_pixel[0]], [target_pixel[1]], c='red', s=40)
    axs[0].set_title("Original + Red Dot")
    axs[0].axis('off')
    if dec==False:
        for i in range(1,5):
            axs[i].imshow(blended_final[i-1])
            axs[i].set_title(f"Layer-{i}")
            axs[i].axis('off')
    else:
        levels = ["p5","p4","p3","p2"]
        for i in range(1,5):
            axs[i].imshow(blended_final[i-1])
            axs[i].set_title(f"Layer-"+levels[i-1])
            axs[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved to {save_path}")


if __name__=='__main__':
    model_names = sorted(name for name in resnet.__all__ if name.islower() and callable(resnet.__dict__[name]))

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='COCOval', choices=['COCO', 'COCOplus', 'ImageNet','COCOval'], help='dataset type')
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
    args.num_instances = 1   
    image_path = "/work/vision_lab/DenseSSL/FPN/SlotCon-main/000000000019.jpg"  # <- replace with your input image
    output_path = "DeConencoderCropped.png"
    feature_level = 0
    model = get_model(args).cuda()
    img_pil, img_tensor = load_image(image_path)

    with torch.no_grad():
        feature = model.encoder_k(img_tensor.cuda(non_blocking=True))#.mean(dim=(-2, -1))
        if args.dec_feature==True:
            feature = model.decoder_k(feature)
            
    #target_pixel = (112, 112)
    target_pixel = (40, 100)
    attn = []
    if args.dec_feature==True:
        levels = ["p5","p4","p3","p2"]
        for level in levels:
            feature_to_work_on = feature[level]#feature["p2"]
            print(feature_to_work_on.shape)
            attn1 = cosine_attention_map(feature_to_work_on.squeeze(0), target_pixel)
            attn.append(attn1)  
        visualize_attention(img_pil, attn, target_pixel, output_path,dec=True)
    else:
        for level in range(0,4):
            feature_to_work_on = feature[level]#feature["p2"]
            attn1 = cosine_attention_map(feature_to_work_on.squeeze(0), target_pixel)
            attn.append(attn1)  
        visualize_attention(img_pil, attn, target_pixel, output_path)



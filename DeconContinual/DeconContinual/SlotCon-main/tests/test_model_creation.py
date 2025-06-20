import sys
sys.path.append("/home/sebquet/scratch/VisionResearchLab/DenseSSL/DenseSSL/SlotCon-main/")

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from data.transforms import CustomDataAugmentation
from models import resnet
from models.slotcon import SlotCon
from main_pretrain import get_parser


from models.resnet import resnet50
import mmseg
from mmengine import Config
from mmengine.runner import Runner
# from mmengine.registry import MODELS

from mim import train
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset

class TestImageFolder(Dataset):
    def __init__(
        self,
        dataset,
        data_dir,
        transform
    ):
        super(TestImageFolder, self).__init__()
        fnames = [
            "/home/sebquet/scratch/VisionResearchLab/DenseSSL/Data/COCO/train2017/000000000009.jpg",
            "/home/sebquet/scratch/VisionResearchLab/DenseSSL/Data/COCO/train2017/000000581929.jpg"
        ]
        self.fnames = np.array(fnames) # to avoid memory leak
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fpath = self.fnames[idx]
        image = Image.open(fpath).convert('RGB')
        return self.transform(image)


    
def get_our_model_and_dataloader(args):

    args.no_ddp = True
    args.world_size = 1


    transform = CustomDataAugmentation(args.image_size, args.min_scale)
    train_dataset = TestImageFolder(args.dataset, args.data_dir, transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False, 
        num_workers=0, pin_memory=True, sampler=None, drop_last=True)
    
    args.num_instances = len(train_loader.dataset)
    
    encoder = resnet.__dict__[args.arch]
    model = SlotCon(encoder, args, use_decoder=True)
  
    return model, train_loader


def get_mmseg_model(config_path):
    # config_path = "/home/sebquet/scratch/VisionResearchLab/DenseSSL/DenseSSL/SlotCon-main/tests/test_config.py"
    cfg = Config.fromfile(config_path)
    
    cfg["work_dir"] = "/home/sebquet/scratch/VisionResearchLab/DenseSSL/tests_mm/"

    # print(f'Config:\n{cfg.pretty_text}')

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    ##  train function is a wrapper that does everything: model creation, dataset creation, starts the training etc. 
    # train(package='mmseg', config=test_config, gpus=0,
    #     other_args=('--work-dir', "/home/sebquet/scratch/VisionResearchLab/DenseSSL/tests_mm/"))\
    
    ## Runner can probably not be used with our version of the software
    # runner = Runner.from_cfg(cfg)
    # # Cannot start a runner with ars if you don't give all args.
    # # runner = Runner(work_dir="/home/sebquet/scratch/VisionResearchLab/DenseSSL/DenseSSL/tests_seb").from_cfg(cfg)

    # # start training
    # # runner.train()
    # mmseg_model = runner.model


    # TODO when you finish downloading the dataset, check creation goes smoothly
    # dataset = build_dataset(cfg.data.train)
    dataset = None
    return model, dataset

def create_config(encoder_ckpt_path, decoder_ckpt_path, config_path):
    config = """model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 2),
        strides=(1, 2, 2, 1),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='{enc_ckpt}'
        )
        ),
    decode_head=dict(
        type='FCNHead',
        in_channels=2048,
        in_index=3,
        channels=256,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0,
        num_classes=19,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        dilation=6,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='{dec_ckpt}'
        )),
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='Resize', img_scale=(2049, 1025), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(769, 769)),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Normalize',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                to_rgb=True),
            dict(type='Pad', size=(769, 769), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ])
)
""".format(enc_ckpt=encoder_ckpt_path, dec_ckpt=decoder_ckpt_path)
    # writing the config in a python file
    assert config_path.endswith(".py")
    with open(config_path, "w") as f:
        f.write(config)
    

def convert_pretrain_to_mm(full_model_path, enc_model_path, dec_model_path):
    obj = torch.load(full_model_path, map_location="cpu")
    if "state_dict" in obj:
        obj = obj["state_dict"]
    elif "model" in obj:
        obj = obj["model"]
    else:
        raise Exception("Cannot find the model in the checkpoint.")
    # if any obj contains module then it is DDP 
    # if not single GPU loading

    # get heyword
    start_kw = []
    for k, v in obj.items():
        param_kw = k.split(".")
        if "encoder_k" in param_kw:
            for i, kw in enumerate(param_kw):
                if kw == "encoder_k":
                    break
                start_kw.append(kw)
            break

    start_kw = ".".join(start_kw)

    new_model = {}
    for k, v in obj.items():
        if not k.startswith(f"{start_kw}encoder_k.") or "fc" in k:
            continue
        old_k = k
        k = k.replace(f"{start_kw}encoder_k.", "")
        print(old_k, "->", k)
        new_model[k] = v
        
    res = {"state_dict": new_model}

    torch.save(res, enc_model_path)


    new_decoder_model = {}      
    for k, v in obj.items():
        if not k.startswith(f"{start_kw}decoder_k.") or "fc" in k:
            continue
        old_k = k
        k = k.replace(f"{start_kw}decoder_k.", "")
        print(old_k, "->", k)
        new_decoder_model[k] = v

  
    res = {"state_dict": new_decoder_model}

    torch.save(res, dec_model_path)


def update_model_params(model):
    """Goes through each parameters of the model and 
    randomly updates the parameters with crazy values"""
    for param in model.parameters():
        param.data = torch.randn_like(param.data)
    return model

if __name__ == "__main__":
    import os
    import inspect 

    project_dir = os.path.dirname(__file__)
    args = get_parser()


    slotcon_model, train_loader = get_our_model_and_dataloader(args)

    # # Update model params
    for i, batch in enumerate(train_loader):
        crops, coords, flags = batch

        # compute output and loss
        loss = slotcon_model((crops, coords, flags))
        print("loss ", loss)
        loss.backward()
        break
    exit(0)
    slotcon_model = update_model_params(slotcon_model)

    our_ckpt_path = os.path.join(project_dir, "test_encoder_decoder_model.pth")
    state = {
        'args': args,
        'model': slotcon_model.state_dict(),
    }
    torch.save(state, our_ckpt_path)

    pretrained_encoder_path = our_ckpt_path.replace("encoder_decoder", "encoder_only")
    pretrained_decoder_path = our_ckpt_path.replace("encoder_decoder", "decoder_only")
    convert_pretrain_to_mm(
        our_ckpt_path, 
        pretrained_encoder_path, 
        pretrained_decoder_path, 
        )
    create_config(
        pretrained_encoder_path, 
        pretrained_decoder_path, 
        config_path=os.path.join(project_dir, "test_config.py")
    )


    mmseg_model, dataset = get_mmseg_model(config_path=os.path.join(project_dir, "test_config.py"))

    # exit()

    # print("input tranform ", mmseg_model.decode_head.input_transform)
    # exit()
    # print(inspect.getsource(mmseg_model.forward_train))
    # print(inspect.getsource(mmseg_model.decode_head))
    # # print("Model created with mmseg ", mmseg_model)
    # exit()
    # print(dataset)
    # y = mmseg_model(dataset.__getitem__(0))
    # print(y)
    # exit()

    # TODO: Compare both models to see if the decoder weights were correctly loaded
    # from torchvision.models import resnet50 as tresnet50
    # resn = tresnet50()
    # print('PYTORHC OFFICIAL resnet')
    # print(resn)
    # exit()
    # resn = resnet50(head_type = 'multi_layer')
    # print("OURS from slotcon")
    # print(resn)
    # print("THEIRS from mmseg   ")
    # print(mmseg_model.backbone)
    for i, batch in enumerate(train_loader):
        crops, coords, flags = batch

        # features = resn(crops[0])
        # print("features len ", len(features))
        # print("shape last features ", features[0].shape)
        # print("shape last features ", features[1].shape)
        # print("shape last features ", features[2].shape)
        # print("shape last features ", features[3].shape)
        # y = mmseg_model.decode_head(features)
        # print(y.shape)
        # exit()
        print("crops shape ", crops[0].shape)
        y = mmseg_model.forward_train(crops[0], 5, 5)
        print("y shape ", y.shape)
        # compute output and loss
        features = slotcon_model.encoder_q(crops[0])
        print("shape features ", features.shape)
        decoded_features = mmseg_model.decode_head([features])
        print("shape decoded features " ,decoded_features.shape)
        break
    
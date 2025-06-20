norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
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
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint=
        #     '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/DenseSSL/SlotCon-main/transfer/segmentation/backbonemodels/city_200_officialweight.pth'
        # )
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
        dilation=6),
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
crop_size = (769, 769)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2049, 1025), ratio_range=(0.5, 2.0)),
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
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2049, 1025),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='CityscapesDataset',
        # data_root='data/cityscapes/',
        data_root = "/home/sebquet/scratch/VisionResearchLab/DenseSSL/Data/COCO/",
        # img_dir='leftImg8bit/train',
        img_dir='train2017',
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
        ]),
    val=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2049, 1025),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2049, 1025),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict()
lr_config = dict(policy='step', step=[63000, 81000], gamma=0.1, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=90000)
checkpoint_config = dict(by_epoch=False, interval=9000)
evaluation = dict(interval=9000, metric='mIoU', pre_eval=True)
work_dir = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/DenseSSL/SlotCon-main/transfer/segmentation/work_dirs/cityscapes/fcn_d6_r50-d16_769x769_90k_cityscapes_moco/city_200_officialweight.pth'
gpu_ids = range(0, 1)
auto_resume = False

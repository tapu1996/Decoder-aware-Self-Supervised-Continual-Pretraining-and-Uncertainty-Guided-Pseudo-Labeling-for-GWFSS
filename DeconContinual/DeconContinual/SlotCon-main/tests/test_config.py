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
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/lustre07/scratch/sebquet/VisionResearchLab/DenseSSL/DenseSSL/SlotCon-main/tests/test_encoder_only_model.pth'
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
            checkpoint='/lustre07/scratch/sebquet/VisionResearchLab/DenseSSL/DenseSSL/SlotCon-main/tests/test_decoder_only_model.pth'
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

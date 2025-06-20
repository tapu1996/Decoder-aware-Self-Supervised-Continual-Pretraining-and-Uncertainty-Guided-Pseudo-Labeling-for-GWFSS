norm_cfg = dict(type='SyncBN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(640, 640)  # match your crop/resize
)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,  # add init_cfg if using pretrained weights

    backbone=dict(
        type='ConvNeXt',
        arch='large',  # ConvNeXt-Large
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        # Optionally include pretrained:
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='https://download.openmmlab.com/.../convnext-large_in1k_20220301.pth',
        #     prefix='backbone.'
        # )
    ),

    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=1536,  # stage 4 output from ConvNeXt-L
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=192,  # stage 0 output from ConvNeXt-L
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),

    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,  # stage 2 output from ConvNeXt-L
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    ),

    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

# data_preprocessor = dict(
#     type='SegDataPreProcessor',
#     mean=[91.267, 94.219, 69.408],
#     std=[59.361, 60.265, 50.824],
#     bgr_to_rgb=True,
#     pad_val=0,
#     seg_pad_val=255)


model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='BEiTv2',
        img_size=512,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        init_values=1e-6,
        drop_path_rate=0.2,
        out_indices=[7, 11, 15, 23],
        ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=512,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))



# model = dict(
#     data_preprocessor=data_preprocessor,
#     pretrained='pretrain/beit_large_patch16_224_pt22k_ft22k.pth',
#     backbone=dict(
#         type='BEiT',
#         embed_dims=1024,
#         num_layers=24,
#         num_heads=16,
#         mlp_ratio=4,
#         qv_bias=True,
#         init_values=1e-6,
#         drop_path_rate=0.2,
#         out_indices=[7, 11, 15, 23]),
#     neck=dict(embed_dim=1024, rescales=[4, 2, 1, 0.5]),
#     decode_head=dict(
#         in_channels=[1024, 1024, 1024, 1024], num_classes=150, channels=1024),
#     auxiliary_head=dict(in_channels=1024, num_classes=150),
#     test_cfg = dict(mode='slide', crop_size=crop_size, stride=(341, 341))
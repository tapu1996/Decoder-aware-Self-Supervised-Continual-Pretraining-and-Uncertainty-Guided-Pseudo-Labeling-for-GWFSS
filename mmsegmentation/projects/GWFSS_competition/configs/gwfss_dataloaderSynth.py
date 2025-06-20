# dataset settings
dataset_type = 'GWFSSDataset'
data_root = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/data/GWFSS_lesstest'  # <-- change this to your actual path
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.8, 1.2),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # New: Random rotation (±10 degrees)
    dict(
        type='RandomRotate',
        prob=0.5,
        degree=10,
        pad_val=0,
        seg_pad_val=255  # or use 0 if no ignored class
    ),

    # New: Color jitter (brightness, contrast, etc.)
    # dict(
    #     type='ColorJitter',
    #     brightness=0.3,
    #     contrast=0.3,
    #     saturation=0.3,
    #     hue=0.1
    # ),

    # # New: Random Gaussian Blur
    # dict(
    #     type='Albu',
    #     transforms=[
    #         dict(type='GaussianBlur', blur_limit=(3, 7), p=0.5),  # ← This is the blur
    #         # You can add other Albumentations here as needed
    #     ],
    #     keymap={'img': 'image', 'gt_semantic_seg': 'mask'},
    #     update_pad_shape=False
    # ),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale_factor=r, keep_ratio=True)
             for r in img_ratios],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ],
            [dict(type='LoadAnnotations')],
            [dict(type='PackSegInputs')]
        ])
]

# train_dataloader = dict(
#     batch_size=12,
#     num_workers=8,
#     persistent_workers=True,
#     sampler=dict(type='InfiniteSampler', shuffle=True),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         reduce_zero_label=False,
#         data_prefix=dict(
#             img_path='img_dir/train', seg_map_path='ann_dir/train'),
#         pipeline=train_pipeline)
# )


train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                reduce_zero_label=False,
                data_prefix=dict(
                    img_path='img_dir/train', seg_map_path='ann_dir/train'),
                pipeline=train_pipeline
            ),
            dict(
                type=dataset_type,
                data_root="/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/data/synth_datasetAll/selected",
                reduce_zero_label=False,
                data_prefix=dict(
                    img_path='img_dir', seg_map_path='cls_dir'),
                pipeline=train_pipeline
            )
        ]
    )
)


val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='img_dir/test', seg_map_path='ann_dir/test'),
        pipeline=test_pipeline)
)

#test_dataloader = val_dataloader

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='img_dir/test', seg_map_path='ann_dir/test'),
        pipeline=tta_pipeline  # ✅ Use TTA pipeline here
    )
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

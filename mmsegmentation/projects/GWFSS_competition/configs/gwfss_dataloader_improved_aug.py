# dataset settings
dataset_type = 'GWFSSDataset'
data_root = '/home/tapotosh/projects/def-farma/tapotosh/DenseSSL_root/mmsegmentation/projects/GWFSS_competition/data/GWFSS'  # <-- change this to your actual path
crop_size = (512, 512)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomMosaic', prob=0.5, img_scale=(512, 512)),
    # dict(
    #     type='RandomCutOut',
    #     prob=0.5,
    #     n_holes=(1, 3),
    #     cutout_ratio=[(0.1, 0.1), (0.2, 0.2)],
    #     fill_in=(0, 0, 0),
    #     seg_fill_in=255
    # ),
    dict(type='RandomResize', scale=(2048, 512), ratio_range=(0.8, 1.2), keep_ratio=True),

    # RandomResize + Mosaic (needs MultiImageMixDataset)
    

    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=10, pad_val=0, seg_pad_val=255),

    Photometric distortion (HSV-like jitter)
    dict(type='PhotoMetricDistortion'),

    # CLAHE contrast enhancement
    dict(type='CLAHE', clip_limit=40.0, tile_grid_size=(8, 8)),

    # Random CutOut
    
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),


    # Gamma adjustment
    #dict(type='AdjustGamma', gamma=1.2),

    # Force all images & masks to fixed size (avoid batching issues)
    # dict(
    #     type='Resize',
    #     scale=(512, 512),
    #     keep_ratio=False
    # ),
        

    # Edge map generation
    #dict(type='GenerateEdge', edge_width=3, ignore_index=255),

   

    

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
#     batch_size=8,
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
        type='MultiImageMixDataset',  # <- must wrap your dataset to support Mosaic
        dataset=dict(
            type='GWFSSDataset',
            data_root=data_root,
            reduce_zero_label=False,
            data_prefix=dict(
                img_path='img_dir/train',
                seg_map_path='ann_dir/train'
            ),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', reduce_zero_label=False),
            ]  # base pipeline used for Mix
        ),
        pipeline=train_pipeline
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

test_dataloader = val_dataloader

# test_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         reduce_zero_label=False,
#         data_prefix=dict(
#             img_path='img_dir/test', seg_map_path='ann_dir/test'),
#         pipeline=tta_pipeline  # ✅ Use TTA pipeline here
#     )
# )

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

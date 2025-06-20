# Optimizer
#optimizer = dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=0.0001)


# optimizer = dict(
#     type="AdamW", lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05
# )

# optimizer = dict(
#     type='AdamW',
#     lr=0.0001,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
# )

# # Optim wrapper config with constructor and paramwise_cfg
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=optimizer,
#     constructor='LearningRateDecayOptimizerConstructor',
#     paramwise_cfg=dict(
#         decay_rate=0.9,
#         decay_type='stage_wise',
#         num_layers=6
#     )
# )


# # optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', type='AdamW', 
# #                  lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
# #                  paramwise_cfg={'decay_rate': 0.9,
# #                                 'decay_type': 'stage_wise',
# #                                 'num_layers': 6})

# lr_config = dict(policy='poly',
#                  warmup='linear',
#                  warmup_iters=1500,
#                  warmup_ratio=1e-6,
#                  power=1.0, min_lr=0.0, by_epoch=False)


optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    constructor='LearningRateDecayOptimizerConstructor',
    paramwise_cfg=dict(
        decay_rate=0.9,
        decay_type='stage_wise',
        num_layers=6
    )
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=1e-5, power=0.9, by_epoch=False, begin=1500, end=40000)
]



# lr_config = dict(
#     _delete_=True,
#     policy="poly",
#     warmup="linear",
#     warmup_iters=1500,
#     warmup_ratio=1e-6,
#     power=1.0,
#     min_lr=0.0,
#     by_epoch=False,
# )

# Learning rate scheduler (step LR policy)
# param_scheduler = [
#     dict(
#         type='MultiStepLR',
#         milestones=[21000, 27000],
#         gamma=0.1,
#         by_epoch=False  # iteration-based
#     )
# ]


# param_scheduler = [
#     dict(
#         type='PolyLR',
#         eta_min=1e-4,
#         power=0.9,
#         begin=0,
#         end=30000,
#         by_epoch=False)
# ]



# Training loop config (40k iterations)
#train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=8000,dynamic_intervals=[(6000, 100)])
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=200)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Default hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=40000,  # save at end only
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=6,
        save_last=True
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

# Evaluation
#val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], pre_eval=True)
#test_evaluator = val_evaluator

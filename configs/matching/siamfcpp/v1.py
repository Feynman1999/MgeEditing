exp_name = 'sar_opt_v1'

# model settings
model = dict(
    type='BasicMatching',
    generator=dict(
        type='SIAMFCPP',
        in_cha=1,
        channels=64,
        loss_cls=dict(type='Focal_loss', alpha = 0.95, gamma = 2),
        loss_bbox=dict(type='IOULoss', loc_loss_type='giou'),
        loss_centerness=dict(type='BCELoss')
    ))

# model training and testing settings
train_cfg = dict(scale = 2)
eval_cfg = dict(metrics=['dis', ])
img_norm_cfg = dict(mean=[0.32, ], std=[0.5, ])

# dataset settings
train_dataset_type = 'MatchFolderDataset'
eval_dataset_type = 'MatchFolderDataset'
test_dataset_type = 'MatchFolderDataset'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='opt',
        flag='grayscale'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='sar',
        flag='grayscale'),
    dict(type='RescaleToZeroOne', keys=['opt', 'sar']),
    dict(type='Resize', keys=['opt'], size=[400, 400], interpolation="area"),
    dict(type='Resize', keys=['sar'], size=[256, 256], interpolation="area"),
    dict(type='Normalize', keys=['opt', 'sar'], to_rgb=False, **img_norm_cfg),  # [-1 ~ 1]
    dict(type='Flip', keys=['opt', 'sar'], flip_ratio=0, direction='horizontal'),
    dict(type='Flip', keys=['opt', 'sar'], flip_ratio=0, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['opt', 'sar'], transpose_ratio=0),
    dict(type='ImageToTensor', keys=['opt', 'sar']),  # [H,W,C] -> [C,H,W]
    dict(type='Collect', keys=['opt', 'sar', 'bbox'])
]

eval_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='opt',
        flag='grayscale'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='sar',
        flag='grayscale'),
    dict(type='RescaleToZeroOne', keys=['opt', 'sar']),
    dict(type='Resize', keys=['opt'], size=[400, 400], interpolation="area"),
    dict(type='Resize', keys=['sar'], size=[256, 256], interpolation="area"),
    dict(type='Normalize', keys=['opt', 'sar'], to_rgb=False, **img_norm_cfg),  # [-1 ~ 1]
    dict(type='ImageToTensor', keys=['opt', 'sar']),  # [H,W,C] -> [C,H,W]
    dict(type='Collect', keys=['opt', 'sar', 'bbox'])
]

test_pipeline = [
    
]

dataroot = "/opt/data/private/datasets"
repeat_times = 1

data = dict(
    # train
    samples_per_gpu=64,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=repeat_times,
        dataset=dict(
            type=train_dataset_type,
            data_path= dataroot + "/stage1",
            opt_folder= "optical",
            sar_folder= "sar",
            file_list_name = "train_random.txt",
            pipeline=train_pipeline)),
    # eval
    eval_samples_per_gpu=1,
    eval_workers_per_gpu=4,
    eval=dict(
        type=eval_dataset_type,
        data_path= dataroot + "/stage1",
        opt_folder= "optical",
        sar_folder= "sar",
        file_list_name = "valid_random.txt",
        pipeline=eval_pipeline,
        mode="eval"),
    # test
    test_samples_per_gpu=1,
    test_workers_per_gpu=4,
    test=dict(
        type=test_dataset_type,
        data_path= dataroot + "/stage1",
        opt_folder= "optical",
        sar_folder= "sar",
        file_list_name = "test.txt",
        pipeline=test_pipeline,
        mode="test"),
)

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=1e-3, betas=(0.9, 0.999)))

# learning policy
total_epochs = 20000 // repeat_times

# hooks
lr_config = dict(policy='Step', step=[total_epochs // 10], gamma=0.7)
checkpoint_config = dict(interval=total_epochs // 5)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='VisualDLLoggerHook')
    ])
visual_config = None
evaluation = dict(interval=200, save_image=False)

# runtime settings
work_dir = f'./workdirs/{exp_name}'
load_from = None
resume_from = None
resume_optim = True
workflow = 'train'

# logger
log_level = 'INFO'

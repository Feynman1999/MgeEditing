# block
exp_name = 'sar_opt_v7'

x_size = 400
z_size = 176
test_z_size = int(800 * z_size / x_size )

# model settings
model = dict(
    type='BasicMatchingV3',
    generator=dict(
        type='SIAMFCPP',
        in_cha=1,
        channels=36,
        loss_cls=dict(type='Focal_loss', alpha = 0.98, gamma = 2),
        loss_bbox=dict(type='IOULoss', loc_loss_type='giou'),
        stacked_convs = 3,
        feat_channels = 36,
        z_size = z_size,
        x_size = x_size,
        test_z_size = test_z_size,
        lambda1 = 0.1,  # reg
        bbox_scale = 0.15,
        stride = 2,
        backbone_type = "alexnet",
        use_distance_cls_label = True
    ))

# model training and testing settings
train_cfg = None
eval_cfg = dict(metrics=['dis', ])
img_norm_cfg = dict(mean=[0.4, ], std=[0.5, ])

# dataset settings
train_dataset_type = 'MatchFolderDataset'
eval_dataset_type = 'MatchFolderDataset'
test_dataset_type = 'MatchFolderDataset'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='opt',
        flag='color'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='sar',
        flag='color'),  # H,W,3  BGR
    dict(type='ColorJitter', keys=['opt', 'sar'], brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0),
    dict(type='Corner_Shelter', keys=['opt'], shelter_ratio = 0, black_ratio=0.75),
    dict(type='Corner_Shelter', keys=['sar'], shelter_ratio = 0, black_ratio=0.75),
    dict(type='Bgr2Gray', keys=['opt', 'sar']),  # H, W, 1
    dict(type='Random_Crop_Opt_Sar', keys=['opt', 'sar'], size=[x_size, z_size]),
    dict(type='RescaleToZeroOne', keys=['opt', 'sar']),
    dict(type='Normalize', keys=['opt', 'sar'], to_rgb=False, **img_norm_cfg),
    dict(type='Flip', keys=['opt', 'sar', 'bbox'], flip_ratio=0.5, direction='horizontal', Len = x_size),
    dict(type='Flip', keys=['opt', 'sar', 'bbox'], flip_ratio=0.5, direction='vertical', Len = x_size),
    dict(type='RandomTransposeHW', keys=['opt', 'sar', 'bbox'], transpose_ratio=0.5),
    dict(type='ImageToTensor', keys=['opt', 'sar']),  # [H,W,C] -> [C,H,W]
    dict(type='Collect', keys=['opt', 'sar', 'bbox', 'class_id', 'file_id'])
]

eval_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='opt',
        flag='color'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='sar',
        flag='color'),  # H,W,3  BGR
    dict(type='Bgr2Gray', keys=['opt', 'sar']),  # H, W, 1
    dict(type='RescaleToZeroOne', keys=['opt', 'sar']),
    dict(type='Normalize', keys=['opt', 'sar'], to_rgb=False, **img_norm_cfg),  
    dict(type='ImageToTensor', keys=['opt', 'sar']),  # [H,W,C] -> [C,H,W]
    dict(type='Collect', keys=['opt', 'sar', 'bbox', 'class_id', 'file_id'])
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='opt',
        flag='color'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='sar',
        flag='color'),  # H,W,3  BGR
    dict(type='Bgr2Gray', keys=['opt', 'sar']),  # H, W, 1
    dict(type='RescaleToZeroOne', keys=['opt', 'sar']),
    dict(type='Normalize', keys=['opt', 'sar'], to_rgb=False, **img_norm_cfg),  
    dict(type='ImageToTensor', keys=['opt', 'sar']),  # [H,W,C] -> [C,H,W]
    dict(type='Collect', keys=['opt', 'sar', 'class_id', 'file_id'])  
]

dataroot = "/opt/data/private/datasets"
repeat_times = 1

data = dict(
    # train
    samples_per_gpu=32,
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
            pipeline=train_pipeline,
            balance_flag = "None")),  # test and uniform and None
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
        data_path= dataroot + "/stage1/test1",
        opt_folder= "optical",
        sar_folder= "sar",
        file_list_name = "img_list_path.txt",
        pipeline=test_pipeline,
        mode="test"),
)

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=0.4 * 1e-3, betas=(0.9, 0.999), weight_decay=2e-6))  # 0.2 * 1e-3

# learning policy
total_epochs = 2000 // repeat_times

# hooks
lr_config = dict(policy='Step', step=[total_epochs // 10], gamma=0.7)
checkpoint_config = dict(interval=20)
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='VisualDLLoggerHook')
    ])
visual_config = None
evaluation = dict(interval=800, save_image=False)

# runtime settings
work_dir = f'./workdirs/{exp_name}'
load_from = f'./workdirs/{exp_name}/20201231_014727/checkpoints/epoch_120'
resume_from = None
resume_optim = True
workflow = 'train'

# logger
log_level = 'INFO'

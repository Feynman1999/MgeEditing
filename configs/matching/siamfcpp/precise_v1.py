# stage 2
exp_name = 'sar_opt_precise_v1'

test_z_size = 512
test_x_size = 520
z_size = 256
x_size = 260

# model settings
model = dict(
    type='PreciseMatching',
    generator=dict(
        type='SIAMFCPP_P',
        in_cha=1,
        channels=56,
        loss_cls=dict(type='Focal_loss', alpha = 0.9, gamma = 2),
        stacked_convs = 2,
        feat_channels = 48,
        z_size = z_size,
        x_size = x_size,
        test_z_size = test_z_size,
        test_x_size = test_x_size,
        backbone_type = "alexnet"  # alexnet  Shuffle_weightnet
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
    dict(type='ColorJitter', keys=['opt', 'sar'], brightness=0.5, contrast=0.5, saturation=0.0, hue=0.0),
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
    dict(type='Random_Crop_Opt_Sar', keys=['opt', 'sar'], size=[test_x_size, test_z_size], have_seed=True),
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
    samples_per_gpu=12,
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
            scale = 1,
            balance_flag = "None")),  # test and uniform and None
    # eval
    eval_samples_per_gpu=1,
    eval_workers_per_gpu=8,
    eval=dict(
        type=eval_dataset_type,
        data_path= dataroot + "/stage1",
        opt_folder= "optical",
        sar_folder= "sar",
        file_list_name = "valid_random.txt",
        pipeline=eval_pipeline,
        mode="eval",
        scale = 1),
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
        mode="test",
        scale = 1),
)

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=0.5 * 1e-3, betas=(0.9, 0.999), weight_decay=2e-5))

# learning policy
total_epochs = 2000 // repeat_times

# hooks
lr_config = dict(policy='Step', step=[total_epochs // 10], gamma=0.7)
checkpoint_config = dict(interval=20)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='VisualDLLoggerHook')
    ])
visual_config = None
evaluation = dict(interval=800, save_image=False)

# runtime settings
work_dir = f'./workdirs/{exp_name}'
load_from = None # f'./workdirs/{exp_name}/20201202_205020/checkpoints/epoch_20' # f'./workdirs/{exp_name}/epoch_70' # f'./workdirs/{exp_name}/20201121_002958/checkpoints/epoch_100'
resume_from = None
resume_optim = True
workflow = 'train'

# logger
log_level = 'INFO'
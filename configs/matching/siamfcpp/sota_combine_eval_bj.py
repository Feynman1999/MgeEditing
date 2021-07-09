exp_name = 'combine_eval'
load_from = "/home/a17/env/MBRVSR/workdirs/epoch_1"
dataroot = "/home/a17/env/dataset"

stage1_z_size = 320
stage1_x_size = 500

stage2_z_size = 256
stage2_x_size = 260

# model settings
model = dict(
    type='TwoStageMatching',
    generator1=dict(
        type='SIAMFCPP_one_sota',
        in_cha=1,
        channels=48,
        loss_cls=dict(type='Focal_loss', alpha = 0.95, gamma = 2),
        loss_bbox=dict(type='IOULoss', loc_loss_type='giou'),
        stacked_convs = 3,
        feat_channels = 48,
        z_size = stage1_z_size,
        x_size = stage1_x_size,
        lambda1 = 0.25,  # reg
        bbox_scale = 0.05,
        stride = 4,
        backbone_type = "alexnet"
    ),
    generator2=dict(
        type='SIAMFCPP_two_sota',
        in_cha=1,
        channels=56,
        stacked_convs = 3,
        feat_channels = 48,
        z_size = stage2_z_size,
        x_size = stage2_x_size,
        test_z_size = 512,
        test_x_size = 520,
        backbone_type = "alexnet"  # alexnet  Shuffle_weightnet
    ))

# model training and testing settings
train_cfg = None
eval_cfg = dict(metrics=['dis', ])
img_norm_cfg = dict(mean=[0.4, ], std=[0.5, ])

# dataset settings
train_dataset_type = 'MatchFolderDataset'
eval_dataset_type = 'MatchFolderDataset'

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
    # dict(type='ColorJitter', keys=['opt', 'sar'], brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0),
    # dict(type='Corner_Shelter', keys=['opt'], shelter_ratio = 0, black_ratio=0.75),
    # dict(type='Corner_Shelter', keys=['sar'], shelter_ratio = 0, black_ratio=0.75),
    dict(type='Bgr2Gray', keys=['opt', 'sar']),  # H, W, 1
    dict(type='Random_Crop_Opt_Sar', keys=['opt', 'sar'], size=[500, 320]),
    dict(type='RescaleToZeroOne', keys=['opt', 'sar']),
    dict(type='Normalize', keys=['opt', 'sar'], to_rgb=False, **img_norm_cfg),
    dict(type='Flip', keys=['opt', 'sar', 'bbox'], flip_ratio=0.5, direction='horizontal', Len = 500),
    dict(type='Flip', keys=['opt', 'sar', 'bbox'], flip_ratio=0.5, direction='vertical', Len = 500),
    dict(type='RandomTransposeHW', keys=['opt', 'sar', 'bbox'], transpose_ratio=0.5),
    dict(type='ImageToTensor', keys=['opt', 'sar']),  # [H,W,C] -> [C,H,W]
    dict(type='Collect', keys=['opt', 'sar', 'bbox'])
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
    # dict(type='Add_contrast', keys=['opt', 'sar'], value_sar = 1, value_optical = 1),
    dict(type='Bgr2Gray', keys=['opt', 'sar']),  # H, W, 1
    dict(type='RescaleToZeroOne', keys=['opt', 'sar']),
    dict(type='Normalize', keys=['opt', 'sar'], to_rgb=False, **img_norm_cfg),  
    dict(type='ImageToTensor', keys=['opt', 'sar']),  # [H,W,C] -> [C,H,W]
    dict(type='Collect', keys=['opt', 'sar', 'bbox', 'class_id', 'file_id'])
]

repeat_times = 1

data = dict(
    # train
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=repeat_times,
        dataset=dict(
            type=train_dataset_type,
            data_path= dataroot,
            opt_folder= "optical",
            sar_folder= "sar",
            file_list_name = "train_random.txt",
            pipeline=train_pipeline,
            balance_flag = "None")),  # test and uniform and None
    # eval
    eval_samples_per_gpu=8,
    eval_workers_per_gpu=1,
    eval=dict(
        type=eval_dataset_type,
        data_path= dataroot,
        opt_folder= "optical",
        sar_folder= "sar",
        file_list_name = "valid_random.txt",
        pipeline=eval_pipeline,
        mode="eval")
)

# optimizer
optimizers = dict(generator1=dict(type='Adam', lr=0.0001 * 1e-3, betas=(0.9, 0.999), weight_decay=2e-6),
                  generator2=dict(type='Adam', lr=0.0001 * 1e-3, betas=(0.9, 0.999), weight_decay=2e-6))

# learning policy
total_epochs = 2000 // repeat_times

# hooks
lr_config = dict(policy='Step', step=[total_epochs // 10], gamma=0.7)
checkpoint_config = dict(interval=total_epochs // 40)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='VisualDLLoggerHook')
    ])
visual_config = None
evaluation = dict(interval=1, save_image=False, multi_process = False)

# runtime settings
work_dir = f'./workdirs/{exp_name}'
resume_from = None 
resume_optim = True
workflow = 'train'

# logger
log_level = 'INFO'

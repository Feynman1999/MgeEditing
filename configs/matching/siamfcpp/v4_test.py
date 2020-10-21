exp_name = 'sar_opt_v4'

ch = 48

# model settings
model = dict(
    type='BasicMatching',
    generator=dict(
        type='SIAMFCPP',
        in_cha=1,
        channels=ch,
        loss_cls=dict(type='Focal_loss', alpha = 0.95, gamma = 2),
        loss_bbox=dict(type='IOULoss', loc_loss_type='giou'),
        loss_centerness=dict(type='BCELoss'),
        feat_channels = ch,
        z_size = 320,
        x_size = 500,
        lambda1 = 0.25,  # reg
        lambda2 = 0.0,  # center
        bbox_scale = 0.05,
        stride = 4
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
    dict(type='ColorJitter', keys=['opt', 'sar'], brightness=0.3, contrast=0.3, saturation=0.3, hue=0.0),
    dict(type='Corner_Shelter', keys=['opt'], shelter_ratio = 0, black_ratio=0.75),
    dict(type='Corner_Shelter', keys=['sar'], shelter_ratio = 0, black_ratio=0.75),
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
            scale = 1,
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
optimizers = dict(generator=dict(type='Adam', lr=0.1 * 1e-3, betas=(0.9, 0.999), weight_decay=2e-4))

# learning policy
total_epochs = 2000 // repeat_times

# hooks
lr_config = dict(policy='Step', step=[total_epochs // 10], gamma=0.7)
checkpoint_config = dict(interval=total_epochs // 40)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='VisualDLLoggerHook')
    ])
visual_config = None
evaluation = dict(interval=400, save_image=False)

# runtime settings
work_dir = f'./workdirs/{exp_name}'
load_from = f'./workdirs/epoch_110'
resume_from = None 
resume_optim = True
workflow = 'test'

# logger
log_level = 'INFO'

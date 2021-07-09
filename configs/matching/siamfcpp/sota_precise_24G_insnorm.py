# stage 2
exp_name = 'sar_opt_precise_v1_insnorm'
dataroot = "/data/home/songtt/chenyuxiang/datasets/stage2"
lr = 0.6 * 1e-3
weight_decay = 2e-6
############################################

test_z_size = 512
test_x_size = 520
z_size = 256
x_size = 260

# model settings
model = dict(
    type='PreciseMatching',
    generator=dict(
        type='SIAMFCPP_two_insnorm', # SIAMFCPP_P
        in_cha=1,
        channels=80,
        stacked_convs = 2,
        feat_channels = 48,
        z_size = z_size,
        x_size = x_size,
        test_z_size = test_z_size,
        test_x_size = test_x_size,
        backbone_type = "alexnet",  # alexnet  Shuffle_weightnet
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
        flag='color'),  # H,W,1
    # dict(type='NLmeanDenoising', keys=['sar'], h=18, kernel =7 , search =25),
    dict(type='ColorJitter', keys=['opt', 'sar'], brightness=0.1, contrast=0.5, saturation=0.0, hue=0.0),
    # dict(type='Corner_Shelter', keys=['opt'], shelter_ratio = 0, black_ratio=0.75),
    # dict(type='Corner_Shelter', keys=['sar'], shelter_ratio = 0, black_ratio=0.75),
    dict(type='Bgr2Gray', keys=['opt', 'sar']),  # H, W, 1
    dict(type='Random_Crop_Opt_Sar', keys=['opt', 'sar'], size=[x_size, z_size]),
    dict(type='RescaleToZeroOne', keys=['opt', 'sar']),
    dict(type='Normalize', keys=['opt', 'sar'], to_rgb=False, **img_norm_cfg),
    dict(type='Flip', keys=['opt', 'sar', 'bbox'], flip_ratio=0.5, direction='horizontal', Len = x_size),
    dict(type='Flip', keys=['opt', 'sar', 'bbox'], flip_ratio=0.5, direction='vertical', Len = x_size),
    dict(type='RandomTransposeHW', keys=['opt', 'sar', 'bbox'], transpose_ratio=0.5),
    dict(type='ImageToTensor', keys=['opt', 'sar']),
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
    # dict(type='NLmeanDenoising', keys=['sar'], h=18, kernel =7 , search =25),
    dict(type='Bgr2Gray', keys=['opt', 'sar']),  # H, W, 1
    dict(type='Random_Crop_Opt_Sar', keys=['opt', 'sar'], size=[test_x_size, test_z_size], have_seed=True),
    dict(type='RescaleToZeroOne', keys=['opt', 'sar']),
    dict(type='Normalize', keys=['opt', 'sar'], to_rgb=False, **img_norm_cfg),  
    dict(type='ImageToTensor', keys=['opt', 'sar']),  # [H,W,C] -> [C,H,W]
    dict(type='Collect', keys=['opt', 'sar', 'bbox', 'class_id', 'file_id'])
]

repeat_times = 1

data = dict(
    # train
    samples_per_gpu=16,
    workers_per_gpu=4,
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
    eval_samples_per_gpu=1,
    eval_workers_per_gpu=4,
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
optimizers = dict(generator=dict(type='AdamW', lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay))

# learning policy
total_epochs = 2000 // repeat_times

# hooks
lr_config = dict(policy='Step', step=[total_epochs // 10], gamma=0.7)
checkpoint_config = dict(interval=10)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook', average_length=10),
        # dict(type='VisualDLLoggerHook')
    ])
visual_config = None
evaluation = dict(interval=800, save_image=False, multi_process=False)

# runtime settings
work_dir = f'./workdirs/{exp_name}'
load_from = f'./workdirs/{exp_name}/20210509_170923/checkpoints/epoch_290' 
resume_from = None
resume_optim = True
workflow = 'train'

# logger
log_level = 'INFO'
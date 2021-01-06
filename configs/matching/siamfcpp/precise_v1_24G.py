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
        type='SIAMFCPP_P', # SIAMFCPP_two_sota  SIAMFCPP_P
        in_cha=1,
        channels=56, #  48 40   80 64
        stacked_convs = 3,
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
        flag='grayscale'),  # H,W,1
    # dict(type='NLmeanDenoising', keys=['sar'], h=18, kernel =7 , search =25),
    dict(type='ColorJitter', keys=['opt', 'sar'], brightness=0.2, contrast=0.8, saturation=0.0, hue=0.0),
    # dict(type='Add_contrast', keys=['opt', 'sar'], value_sar = 1.5, value_optical = 0.75),
    # dict(type='Corner_Shelter', keys=['opt'], shelter_ratio = 0, black_ratio=0.75),
    # dict(type='Corner_Shelter', keys=['sar'], shelter_ratio = 0, black_ratio=0.75),
    # dict(type='Bgr2Gray', keys=['opt', 'sar']),  # H, W, 1
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
        flag='grayscale'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='sar',
        flag='grayscale'),  # H,W,3  BGR
    # dict(type="Add_brightness", keys=['opt', 'sar'], value_sar = 0.8, value_optical = 1.25),
    dict(type='Add_contrast', keys=['opt', 'sar'], value_sar = 1.5, value_optical = 1),
    # dict(type='NLmeanDenoising', keys=['sar'], h=18, kernel =7 , search =25),
    # dict(type='Bgr2Gray', keys=['opt', 'sar']),  # H, W, 1
    dict(type='Random_Crop_Opt_Sar', keys=['opt', 'sar'], size=[test_x_size, test_z_size], have_seed=True),
    dict(type='RescaleToZeroOne', keys=['opt', 'sar']),
    dict(type='Normalize', keys=['opt', 'sar'], to_rgb=False, **img_norm_cfg),
    dict(type='Flip', keys=['opt', 'sar', 'bbox'], flip_ratio=0.5, direction='horizontal', Len = test_x_size),
    dict(type='Flip', keys=['opt', 'sar', 'bbox'], flip_ratio=0.5, direction='vertical', Len = test_x_size),
    dict(type='RandomTransposeHW', keys=['opt', 'sar', 'bbox'], transpose_ratio=0.5),
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

dataroot = "/data/home/songtt/work/datasets"
repeat_times = 1

data = dict(
    # train
    samples_per_gpu=2,
    workers_per_gpu=4,
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
            balance_flag = "uniform")),  # test and uniform and None
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
optimizers = dict(generator=dict(type='Adam', lr=0.00001 * 1e-3, betas=(0.9, 0.999), weight_decay=2e-5)) # 1 -> 0.4 

# learning policy
total_epochs = 2000 // repeat_times

# hooks
lr_config = dict(policy='Step', step=[total_epochs // 10], gamma=0.7)
checkpoint_config = dict(interval=10)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='VisualDLLoggerHook')
    ])
visual_config = None
evaluation = dict(interval=1, save_image=False)

# runtime settings
work_dir = f'./workdirs/{exp_name}'
load_from = f'./workdirs/sota/epoch_1' # f'./workdirs/{exp_name}/20210106_152419/checkpoints/epoch_100' # f'./workdirs/{exp_name}/20210101_225424/checkpoints/epoch_20' # f'./workdirs/{exp_name}/epoch_70' # f'./workdirs/{exp_name}/20201121_002958/checkpoints/epoch_100'
resume_from = None
resume_optim = True
workflow = 'train'

# logger
log_level = 'INFO'

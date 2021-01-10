exp_name = 'combine_test_24G'

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
test_dataset_type = 'MatchFolderDataset'

test_pipeline = [
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
    dict(type='Add_contrast', keys=['opt', 'sar'], value_sar = 0.7, value_optical = 0.7),  
    dict(type='RescaleToZeroOne', keys=['opt', 'sar']),
    dict(type='Normalize', keys=['opt', 'sar'], to_rgb=False, **img_norm_cfg),  
    dict(type='ImageToTensor', keys=['opt', 'sar']),  # [H,W,C] -> [C,H,W]
    dict(type='Collect', keys=['opt', 'sar', 'class_id', 'file_id'])  
]

dataroot = "/data/home/songtt/work/datasets"
repeat_times = 1

data = dict(
    test_samples_per_gpu=8,
    test_workers_per_gpu=8,
    test=dict(
        type=test_dataset_type,
        data_path= dataroot + "/test2",
        opt_folder= "optical",
        sar_folder= "sar",
        file_list_name = "img_list_path.txt",
        pipeline=test_pipeline,
        mode="test"),
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
evaluation = dict(interval=1, save_image=False)

# runtime settings
work_dir = f'./workdirs/{exp_name}'
load_from = f'./workdirs/sota/epoch_1'
resume_from = None 
resume_optim = True
workflow = 'test'

# logger
log_level = 'INFO'

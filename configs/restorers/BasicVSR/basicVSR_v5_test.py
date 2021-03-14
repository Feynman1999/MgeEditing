exp_name = 'basicVSR_v5_test'

scale = 4

# model settings
model = dict(
    type='BidirectionalRestorer_small',
    generator=dict(
        type='BasicVSR_v5',
        in_channels=3,
        out_channels=3,
        hidden_channels = 8,
        upscale_factor = scale),
        pixel_loss=dict(type='L2Loss')) # L2Loss CharbonnierLoss

# model training and testing settings
train_cfg = None
eval_cfg = dict(metrics=['PSNR'], crop_border=0, multi_pad = 1, gap = 1)
img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1])

test_dataset_type = 'SRManyToManyDataset'
test_pipeline = [
    dict(type='GenerateFrameIndiceswithPadding', padding="reflection", many2many = False, index_start = 0, name_padding = True),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='Normalize', keys=['lq'], to_rgb=True, **img_norm_cfg),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq', 'num_input_frames', 'LRkey', 'lq_path'])
]

dataroot = "/work_base/datasets/REDS/test"
repeat_times = 1
data = dict(
    test_samples_per_gpu=10,
    test_workers_per_gpu=5,
    test=dict(
        type=test_dataset_type,
        lq_folder= dataroot + "/test_sharp_bicubic/X4",
        num_input_frames=1,
        pipeline=test_pipeline,
        scale=scale,
        mode="test")
)

# optimizer
optimizers = dict(generator=dict(type='SGD', lr=0.00000000000001)) # 2_23  1.5_12  1_10   sgd with momentum 搜最好结果，eval iter1 并且每次都保存
# batch 8 8 1
# learning policy
total_epochs = 400 // repeat_times

# hooks
lr_config = dict(policy='Step', step=[total_epochs // 10], gamma=0.7)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook', average_length=100),
        # dict(type='VisualDLLoggerHook')
    ])
evaluation = dict(interval=1, save_image=False, multi_process=False, ensemble=False)

# runtime settings
work_dir = f'./workdirs/{exp_name}'
load_from = './workdirs/basicVSR_last_v5/20210313_055329/checkpoints/epoch_12'   # 提高一次学习率之后的
resume_from = None
resume_optim = True
workflow = 'test'

# logger
log_level = 'INFO'

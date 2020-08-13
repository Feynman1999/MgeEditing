exp_name = 'fsrcnn_x2_div2k'

scale = 4
# model settings
model = dict(
    type='BasicRestorer',
    generator=dict(
        type='FSRCNN',
        in_channels=3,
        out_channels=3,
        upscale_factor=scale),
    pixel_loss=dict(type='L1Loss'))
# model training and testing settings
train_cfg = None
eval_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=scale)
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])
# dataset settings
train_dataset_type = 'SRFolderDataset'
eval_dataset_type = 'SRFolderDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='PairedRandomCrop', gt_patch_size=128),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize', keys=['lq', 'gt'], to_rgb=True, **img_norm_cfg),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'])
]
eval_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize', keys=['lq', 'gt'], to_rgb=True, **img_norm_cfg),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'])
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
            lq_folder=dataroot + "/DIV2K/DIV2K_train_LR_bicubic/X4",
            gt_folder=dataroot + "/DIV2K/DIV2K_train_HR",
            pipeline=train_pipeline,
            scale=scale,
            filename_tmpl='{}x4')),
    # eval
    eval_samples_per_gpu=1,
    eval_workers_per_gpu=4,
    eval=dict(
        type=eval_dataset_type,
        lq_folder=dataroot + "/Set5/LRbicx4",
        gt_folder=dataroot + "/Set5/GTmod12",
        pipeline=eval_pipeline,
        scale=scale,
        filename_tmpl='{}'),
    # test
    test_samples_per_gpu=1,
    test_workers_per_gpu=4,
    test=dict(
        type=eval_pipeline,
        lq_folder=dataroot + "/Set5/LRbicx4",
        gt_folder=dataroot + "/Set5/GTmod12",
        pipeline=eval_pipeline,
        scale=scale,
        filename_tmpl='{}'))

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))

# learning policy
total_epochs = 2000 // repeat_times

# hooks
lr_config = dict(policy='Step', step=[total_epochs // 10], gamma=0.7)
checkpoint_config = dict(interval=total_epochs // 10)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='VisualDLLoggerHook')
    ])
visual_config = None
evaluation = dict(interval=500, save_image=True)

# runtime settings
work_dir = f'./workdirs/{exp_name}'
load_from = None
resume_from = None
resume_optim = True
workflow = [('train', 1)]

# logger
log_level = 'INFO'

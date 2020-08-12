exp_name = 'dbpn_x4_300k_div2k'

scale = 4
# model settings
model = dict(
    type='BasicRestorer',
    generator=dict(
        type='DBPN',
        in_channels=3,
        out_channels=3,
        n_0=256,
        n_R=64,
        iterations_num=5,
        upscale_factor=scale),
    pixel_loss=dict(type='L1Loss', reduction='mean'))
# model training and testing settings
train_cfg = None
eval_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=scale)
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# dataset settings
train_dataset_type = 'SRFolderDataset'
val_dataset_type = 'SRFolderDataset'
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
    dict(type='PairedRandomCrop', gt_patch_size=40*scale),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize', keys=['lq', 'gt'], to_rgb=True, **img_norm_cfg),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'])
]
test_pipeline = [
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

data = dict(
    # train
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder=dataroot + "/DIV2K/DIV2K_train_LR",
            gt_folder=dataroot + "/DIV2K/DIV2K_train_HR",
            pipeline=train_pipeline,
            scale=scale)),
    # val
    val_samples_per_gpu=1,
    val_workers_per_gpu=4,
    val=dict(
        type=val_dataset_type,
        lq_folder=dataroot + "/Set5/LR",
        gt_folder=dataroot + "/Set5/HR",
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}x4'),
    # test
    test=dict(
        type=val_dataset_type,
        lq_folder=dataroot + "/Set5/LR",
        gt_folder=dataroot + "/Set5/HR",
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}x4'))

# optimizer
optimizers = dict(generator=dict(type='AdamOptimizer', learning_rate=2e-4, beta1=0.9, beta2=0.999))

# learning policy
total_iters = 200000

# hooks
lr_config = dict(policy='Step', by_epoch=False, step=[200000], gamma=0.5)
checkpoint_config = dict(interval=total_iters//20, by_epoch=False)
log_config = dict(
    interval=total_iters // 5000,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='VisualDLLoggerHook')
    ])
visual_config = None
evaluation = dict(interval=total_iters // 100, save_image=False)

# runtime settings
work_dir = f'./workdirs/{exp_name}'
load_from = f"./workdirs/{exp_name}/checkpoint/iter_10000"
resume_from = None
resume_optim = True
workflow = [('train', 1)]

# logger
log_level = 'INFO'

"""
use transformer

train:  select 6 frames 1+3+2

eval and test:   both neighbor and distant are used(3 + x), and some frames are averaged many times

"""
exp_name = 'sttn_v1'

scale = 4

# model settings
model = dict(
    type='STTNRestorer',
    generator=dict(
        type='STTN',
        in_channels=3,
        out_channels=3,
        channels= 80,
        layers = 4,
        heads = 2,
        upscale_factor = scale,
        layer_norm = False),
    pixel_loss=dict(type='CharbonnierLoss', reduction="sum"))

# model training and testing settings
train_cfg = None
eval_cfg = dict(metrics=['PSNR'], crop_border=0)
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])

# dataset settings
train_dataset_type = 'SRManyToManyDataset'
eval_dataset_type = 'SRManyToManyDataset'
test_dataset_type = 'SRManyToManyDataset'

train_pipeline = [
    dict(type='STTN_REDS_GenerateFrameIndices', interval_list=[1, 2], gap = 20),
    # 加入color jitter
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='PairedRandomCrop', gt_patch_size=[90 * 4, 160 * 4]),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize', keys=['lq', 'gt'], to_rgb=True, **img_norm_cfg),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt', 'lq_path', 'gt_path'])
]

eval_pipeline = [
    dict(type='GenerateFrameIndiceswithPadding', padding="reflection", many2many = False, index_start = 0, name_padding = True, dist_gap = 33),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize', keys=['lq', 'gt'], to_rgb=True, **img_norm_cfg),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt', 'num_input_frames', 'LRkey', 'lq_path'])
]

dataroot = "/data/home/songtt/chenyuxiang/datasets/REDS/train"
repeat_times = 1
eval_part =  ('000', '011', '015', '020')  # tuple(map(str, range(240,242)))
data = dict(
    # train
    samples_per_gpu=2,
    workers_per_gpu=6,
    train=dict(
        type='RepeatDataset',
        times=repeat_times,
        dataset=dict(
            type=train_dataset_type,
            lq_folder= dataroot + "/train_sharp_bicubic/X4",
            gt_folder= dataroot + "/train_sharp",
            num_input_frames=3,
            pipeline=train_pipeline,
            scale=scale,
            eval_part = eval_part)),
    # eval
    eval_samples_per_gpu=1,
    eval_workers_per_gpu=4,
    eval=dict(
        type=eval_dataset_type,
        lq_folder= dataroot + "/train_sharp_bicubic/X4",
        gt_folder= dataroot + "/train_sharp",
        num_input_frames=3,
        pipeline=eval_pipeline,
        scale=scale,
        mode="eval",
        eval_part = eval_part)
)

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=2 * 1e-4, betas=(0.9, 0.999))) # 0.5 -> 0.05

# learning policy
total_epochs = 400 // repeat_times

# hooks
lr_config = dict(policy='Step', step=[total_epochs // 10], gamma=0.7)
checkpoint_config = dict(interval=3)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', average_length=50),
        # dict(type='VisualDLLoggerHook')
    ])
evaluation = dict(interval=2000, save_image=False, multi_process=False, ensemble=False)

# runtime settings
work_dir = f'./workdirs/{exp_name}'
load_from = None # f'./workdirs/{exp_name}/20210120_191641/checkpoints/epoch_42'
resume_from = None
resume_optim = True
workflow = 'train'

# logger
log_level = 'INFO'

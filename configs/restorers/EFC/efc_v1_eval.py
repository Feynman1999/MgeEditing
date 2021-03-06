"""
Every Frame Counts - Content-Aware video super resolution with transformer
目标： PSNR on REDS4  30.5

block size: 9*8  for 180*320 (REDS dataset)

training: (REDS dataset)
0. pre-deal optical flow for all training dataset

1. Select sub frames and sub class area(slic method) with different params(e.g. 30 in 100 frames, and n_segments or compactness)

2. forward and backward training

eval / test: 
0. pre-deal optical flow for all eval / test dataset

1. Select the same class area as long as possible (blocks), and to make sure that all blocks in 100 frames will be selected at least once

2. forward

3. Average the results by blocks
"""
exp_name = 'efc_baseline'

scale = 4

# model settings
model = dict(
    type='EFCRestorer', #  EFCRestorer
    generator=dict(
        type='EFC', # STTN  EFC
        in_channels=3,
        out_channels=3,
        channels = 16,
        layers = 5,
        heads = 4,
        upscale_factor = scale,
        layer_norm = False),
    pixel_loss=dict(type='CharbonnierLoss', reduction="mean"))

# model training and testing settings
train_cfg = None
eval_cfg = dict(metrics=['PSNR'], crop_border=0, multi_pad = 1, gap = 1)
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])

# dataset settings
train_dataset_type = 'SRManyToManyDataset'
eval_dataset_type = 'SRManyToManyDataset'

train_pipeline = [
    dict(type='STTN_REDS_GenerateFrameIndices', interval_list=[1], gap = 0),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        flag='unchanged',
        make_bin=False),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        flag='unchanged',
        make_bin=False),
    # dict(type='PairedRandomCrop', gt_patch_size=[108 * 4, 192 * 4]),
    dict(type='MinimumBoundingBox_ByOpticalFlow', blocksizes = [9, 8], n_segments = (80, 120), compactness = (10, 20)),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize', keys=['lq', 'gt'], to_rgb=True, **img_norm_cfg),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt'], do_not_stack=True),
    dict(type='Collect', keys=['lq', 'gt', 'lq_path', 'gt_path', 'transpose'])
]

eval_pipeline = [
    dict(type='GenerateFrameIndiceswithPadding', padding="reflection", many2many = False, index_start = 0, name_padding = True),
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

dataroot = "/mnt/tmp/REDS/train"
repeat_times = 1
eval_part =  ('000', '011', '015', '020')  # tuple(map(str, range(240,242)))
data = dict(
    # train
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=repeat_times,
        dataset=dict(
            type=train_dataset_type,
            lq_folder= dataroot + "/train_sharp_bicubic/X4",
            gt_folder= dataroot + "/train_sharp",
            num_input_frames=19,
            pipeline=train_pipeline,
            scale=scale,
            eval_part = eval_part)),
    # eval
    eval_samples_per_gpu=10,
    eval_workers_per_gpu=4,
    eval=dict(
        type=eval_dataset_type,
        lq_folder= dataroot + "/train_sharp_bicubic/X4",
        gt_folder= dataroot + "/train_sharp",
        num_input_frames=1,
        pipeline=eval_pipeline,
        scale=scale,
        mode="eval",
        eval_part = eval_part)
)

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=0.00000000001, betas=(0.9, 0.999)))

# learning policy
total_epochs = 400 // repeat_times

# hooks
lr_config = dict(policy='Step', step=[total_epochs // 10], gamma=0.7)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=3,
    hooks=[
        dict(type='TextLoggerHook', average_length=2000),
        # dict(type='VisualDLLoggerHook')
    ])
evaluation = dict(interval=1, save_image=True, multi_process=False, ensemble=False)

# runtime settings
work_dir = f'./workdirs/{exp_name}'
load_from = f'./workdirs/efc_baseline/20210326_105542/checkpoints/epoch_2'
resume_from = None
resume_optim = True
workflow = 'train'

# logger
log_level = 'INFO'

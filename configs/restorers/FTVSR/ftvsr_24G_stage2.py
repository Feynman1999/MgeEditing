"""
flow based transformer for VSR
"""
load_from = "./workdirs/optical_flow_transformer/20210426_014054/checkpoints/epoch_4"
path2spynet = "./workdirs/spynet/spynet-sintel-final.mge"
dataroot = "/data/home/songtt/chenyuxiang/datasets/REDS/train"
exp_name = 'optical_flow_transformer'
samples_per_gpu = 4

num_input_frames = 15
channels = 8
layers = 6
heads = 4
keept = 2
learned_offsets_num = 2
reconstruction_blocks = 4
flownet_layers = 4
flow_lr_mult = 0.1
offset_lr_mult = 0.1

eval_gap = 2216000 # iter
log_gap = 5 # iter

# you can custom values before, for the following params do not change if you are new to this project
###########################################################################################

learning_rate_per_batch = 2e-4
crop_size = [45*4, 80*4]
scale = 4

# model settings
model = dict(
    type='FTVSRRestorer',
    generator=dict(
        type='FTVSR',
        in_channels=3,
        out_channels=3,
        channels = channels,
        layers = layers,
        reconstruction_blocks = reconstruction_blocks,
        flownet_layers = flownet_layers,
        pretrained_optical_flow_path = path2spynet,
        heads = heads,
        upscale_factor = scale,
        keept = keept,
        learned_offsets_num = learned_offsets_num,
        use_flow_mask = True),
    pixel_loss=dict(type='CharbonnierLoss', reduction="mean"),
    Fidelity_loss = None
)

# model training and testing settings
img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1])
train_cfg = dict(img_norm_cfg = img_norm_cfg, train_cfg = dict(viz_flow = False))
eval_cfg = dict(metrics=['PSNR'], crop_border=0, multi_pad = 1, gap = 1)

# dataset settings
train_dataset_type = 'SRManyToManyDataset'
eval_dataset_type = 'SRManyToManyDataset'  # 统一都用这个dataset

train_pipeline = [
    dict(type='GenerateFrameIndices', interval_list=[1], many2many = True, index_start = 0, name_padding = True, load_flow = False),
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
    dict(type='PairedRandomCrop', gt_patch_size=crop_size, crop_flow=False),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize', keys=['lq', 'gt'], to_rgb=True, **img_norm_cfg),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt', 'lq_path', 'gt_path'])
]

eval_pipeline = [
    dict(type='GenerateFrameIndiceswithPadding', padding="reflection", many2many = False, index_start = 0, name_padding = True),
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
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize', keys=['lq', 'gt'], to_rgb=True, **img_norm_cfg),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt', 'num_input_frames', 'LRkey', 'lq_path'])
]

repeat_times = 1
eval_part = ('000', '011', '015', '020')
data = dict(
    # train
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=samples_per_gpu,
    train=dict(
        type='RepeatDataset',
        times=repeat_times,
        dataset=dict(
            type=train_dataset_type,
            lq_folder= dataroot + "/train_sharp_bicubic/X4",
            gt_folder= dataroot + "/train_sharp",
            num_input_frames=num_input_frames,
            pipeline=train_pipeline,
            scale=scale,
            eval_part = eval_part)),
    # eval
    eval_samples_per_gpu=10,
    eval_workers_per_gpu=5,
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
optimizers = dict(generator=dict(type='Adam', lr=learning_rate_per_batch * samples_per_gpu, betas=(0.9, 0.99),
                                paramwise_cfg=dict(custom_keys={
                                                    'flownet': dict(lr_mult=flow_lr_mult),
                                                    'offsetnet': dict(lr_mult=offset_lr_mult)})))

# learning policy
total_epochs = 400 // repeat_times

# hooks
lr_config = dict(policy='Step', step=[total_epochs // 10], gamma=0.7)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=log_gap,
    hooks=[
        dict(type='TextLoggerHook', average_length=500),
    ])
evaluation = dict(interval=eval_gap, save_image=False, multi_process=False, ensemble=False)

# runtime settings
work_dir = f'./workdirs/{exp_name}'
resume_from = None
resume_optim = True
workflow = 'train'

# logger
log_level = 'INFO'

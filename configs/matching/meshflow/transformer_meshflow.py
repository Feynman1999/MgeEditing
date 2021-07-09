'''
    训练meshflow时，采用90*160的图片，分块大小为9*16
    数据集为reds，每次采样三帧，gap为[1,2,3,4,5]均匀采样
    
    测试使用180*320
'''
dataroot = "/mnt/tmp/REDS/train"  #  "/work_base/datasets/REDS/train"
exp_name = 'transformer_meshflow_v1'

scale = 4

# model settings
model = dict(
    type='MeshFlowMatching',
    generator=dict(
        type='TMF',
        in_channels=3,
        out_channels=2,
        channels = 32,
        layers = 8,
        heads = 4,
        layer_norm = True)
)

# model training and testing settings
train_cfg = None
eval_cfg = dict(metrics=['PSNR'], crop_border=0, multi_pad = 1)
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])

# dataset settings
train_dataset_type = 'SRManyToManyDataset'
eval_dataset_type = 'SRManyToManyDataset'  # 统一都用这个dataset

train_pipeline = [
    dict(type='GenerateFrameIndices', interval_list=[1], many2many = False, index_start = 0, name_padding = True, load_flow = False),
    # dict(type='TemporalReverse', keys=['lq_path'], reverse_ratio=0.5),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        flag='unchanged',
        make_bin=True),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        flag='unchanged',
        make_bin=True),
    dict(type='PairedRandomCrop', gt_patch_size=[90 * 4, 160 * 4], crop_flow=False),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='Normalize', keys=['lq'], to_rgb=True, **img_norm_cfg),
    dict(type='Flip', keys=['lq'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['lq'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq'], transpose_ratio=0),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq', 'lq_path'])
]

eval_pipeline = [
    dict(type='GenerateFrameIndiceswithPadding', padding="reflection", many2many = False, index_start = 0, name_padding = True),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        flag='unchanged',
        make_bin=True),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='Normalize', keys=['lq'], to_rgb=True, **img_norm_cfg),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq', 'num_input_frames', 'LRkey', 'lq_path'])
]

repeat_times = 100
eval_part =  ('000', '011', '015', '020')
data = dict(
    # train
    samples_per_gpu=12,
    workers_per_gpu=12,
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
optimizers = dict(generator=dict(type='Adam', lr=4 * 1e-4, betas=(0.9, 0.999)))

# learning policy
total_epochs = 400 // repeat_times

# hooks
lr_config = dict(policy='Step', step=[total_epochs // 10], gamma=0.7)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=3,
    hooks=[
        dict(type='TextLoggerHook', average_length=100),
        # dict(type='VisualDLLoggerHook')
    ])
evaluation = dict(interval=600000000, save_image=False, multi_process=False, ensemble=False)

# runtime settings
work_dir = f'./workdirs/{exp_name}'
load_from = None
resume_from = None
resume_optim = True
workflow = 'train'

# logger
log_level = 'INFO'

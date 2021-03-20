'''
训练meshflow时，采用54 * 96的图片，采样相隔1，3，5，7，9，11帧的情况
分的格子数量为
大level:  36*64 网格
中level:  18*32  网格
小level:  6*10   网格
最后：统一resize到37 * 65的结果
在basicVSR中，使用相邻1，5，10帧的信息的hidden（训练和测试，训练时对使用5和10帧的信息的帧，loss加权）
'''
exp_name = 'meshflow_v1'

scale = 4

# model settings
model = dict(
    type='MeshFlowMatching',
    generator=dict(
        type='DeepMeshFlow',
        in_channels=3,
        channels = 96,
        init_nums = 3, # 3
        blocknums = 18,
        blocktype = "resblock")
)

# model training and testing settings
train_cfg = None
eval_cfg = dict(metrics=['PSNR'], crop_border=0, multi_pad = 1)
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])

# dataset settings
train_dataset_type = 'SRManyToManyDataset'
eval_dataset_type = 'SRManyToManyDataset'  # 统一都用这个dataset
test_dataset_type = 'SRManyToManyDataset'

train_pipeline = [
    dict(type='GenerateFrameIndices', interval_list=[1, 3, 5, 7, 9, 11], many2many = False, index_start = 0, name_padding = True, load_flow = False),
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
    dict(type='PairedRandomCrop', gt_patch_size=[54 * 4, 96 * 4], crop_flow=False),
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
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='Normalize', keys=['lq'], to_rgb=True, **img_norm_cfg),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq', 'num_input_frames', 'LRkey', 'lq_path'])
]

dataroot = "/work_base/datasets/REDS/train"  #  "/work_base/datasets/REDS/train"
repeat_times = 1
eval_part = tuple(map(str, range(240,270)))
data = dict(
    # train
    samples_per_gpu=8,
    workers_per_gpu=8,
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
optimizers = dict(generator=dict(type='Adam', lr=4 * 1e-4, betas=(0.9, 0.999),
                                paramwise_cfg=dict(custom_keys={
                                                    'flownet': dict(lr_mult=0)})))

# learning policy
total_epochs = 400 // repeat_times

# hooks
lr_config = dict(policy='Step', step=[total_epochs // 10], gamma=0.7)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=3,
    hooks=[
        dict(type='TextLoggerHook', average_length=50),
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

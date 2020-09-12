exp_name = 'rsdn_v5'

scale = 4

# model settings
model = dict(
    type='MOMM',
    generator=dict(
        type='RSDNV2',
        in_channels=3,
        out_channels=3,
        mid_channels=172,
        hidden_channels = 86,
        ch = 43,
        blocknums = 8,
        upscale_factor = scale),
    pixel_loss=dict(type='RSDNLoss'))

# model training and testing settings
train_cfg = None
eval_cfg = dict(metrics=['PSNR'], crop_border=0, padding_multi = 4)
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])

# dataset settings
train_dataset_type = 'SRManyToManyDataset'
eval_dataset_type = 'SRManyToOneDataset'
test_dataset_type = 'SRManyToOneDataset'

train_pipeline = [
    dict(type='GenerateFrameIndices', interval_list=[1], many2many = True, name_padding = True),
    dict(type='TemporalReverse', keys=['lq_path', 'gt_path'], reverse_ratio=0.3),
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
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize', keys=['lq', 'gt'], to_rgb=True, **img_norm_cfg),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'])
]

eval_pipeline = [
    dict(type="GenerateFrameIndiceswithPadding", padding='reflection_circle', name_padding = True),
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
    dict(type='FramesToTensor', keys=['lq', 'gt']), # HWC -> CHW
    dict(type='Collect', keys=['lq', 'is_first', 'gt'])
]

test_pipeline = [
    dict(type="GenerateFrameIndiceswithPadding", padding='reflection_circle', name_padding = True),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='Normalize', keys=['lq'], to_rgb=True, **img_norm_cfg),
    dict(type='FramesToTensor', keys=['lq']), # HWC -> CHW
    dict(type='Collect', keys=['lq', 'is_first'])
]

dataroot = "/home/megstudio/dataset"
repeat_times = 1
eval_part = None
data = dict(
    # train
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=repeat_times,
        dataset=dict(
            type=train_dataset_type,
            lq_folder= dataroot + "/game1/train_png",
            gt_folder= dataroot + "/game1/train_png",
            num_input_frames=9,
            pipeline=train_pipeline,
            scale=scale,
            eval_part = eval_part,
            mode = "train",
            LR_symbol = "_down4x.mp4")),
    # eval
    eval_samples_per_gpu=1,
    eval_workers_per_gpu=4,
    eval=dict(
        type=eval_dataset_type,
        lq_folder= dataroot + "/game1/train_png",
        gt_folder= dataroot + "/game1/train_png",
        num_input_frames = 5,
        pipeline=eval_pipeline,
        scale=scale,
        mode="eval",
        eval_part = eval_part,
        LR_symbol = "_down4x.mp4"),
    # test
    test_samples_per_gpu=1,
    test_workers_per_gpu=4,
    test=dict(
        type=test_dataset_type,
        lq_folder= "/home/megstudio/workspace/test/test1",
        num_input_frames = 5,
        pipeline=test_pipeline,
        scale=scale,
        mode="test")
)

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))

# learning policy
total_epochs = 100 // repeat_times

# hooks
lr_config = dict(policy='Step', step=[total_epochs // 10], gamma=0.7)
checkpoint_config = dict(interval=total_epochs // 50)
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='VisualDLLoggerHook')
    ])
visual_config = None
evaluation = dict(interval=2000000, save_image=True)

# runtime settings
work_dir = f'./workdirs/{exp_name}'
load_from = None
resume_from = f'./workdirs/{exp_name}/20200911_075509/epoch_2'
resume_optim = True
workflow = 'train'

# logger
log_level = 'INFO'

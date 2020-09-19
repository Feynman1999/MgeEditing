exp_name = 'mucan_v2_test'

scale = 4
frames = 9

# model settings
model = dict(
    type='ManytoOneRestorer',
    generator=dict(
        type='MUCAN',
        ch=180,
        nframes = frames,
        input_nc = 3,
        output_nc = 3,
        upscale_factor = scale),
    pixel_loss=dict(type='L1Loss'))

# model training and testing settings
train_cfg = None
eval_cfg = dict(metrics=['PSNR'], crop_border=0, padding_multi = 4)
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])

# dataset settings
train_dataset_type = 'SRManyToOneDataset'
eval_dataset_type = 'SRManyToOneDataset'
test_dataset_type = 'SRManyToOneDataset'

train_pipeline = [
    dict(type='GenerateFrameIndices', interval_list=[1, 2], many2many = False, name_padding = True),
    dict(type='TemporalReverse', keys=['lq_path', 'gt_path'], reverse_ratio=0.2),
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
    dict(type='PairedRandomCrop', gt_patch_size=68 * 4),
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
    dict(type='Collect', keys=['lq', 'gt'])
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
    dict(type='Collect', keys=['lq'])
]


dataroot = "/home/aistudio/work/datasets"
repeat_times = 1
eval_part = ("26", )
data = dict(
    # train
    samples_per_gpu=8,
    workers_per_gpu=3,
    train=dict(
        type='RepeatDataset',
        times=repeat_times,
        dataset=dict(
            type=train_dataset_type,
            lq_folder= dataroot + "/mge/train/LR",
            gt_folder= dataroot + "/mge/train/HR",
            num_input_frames=frames,
            pipeline=train_pipeline,
            scale=scale,
            eval_part = eval_part)),
    # eval
    eval_samples_per_gpu=1,
    eval_workers_per_gpu=0,
    eval=dict(
        type=eval_dataset_type,
        lq_folder= dataroot + "/mge/train/LR",
        gt_folder= dataroot + "/mge/train/HR",
        num_input_frames = frames,
        pipeline=eval_pipeline,
        scale=scale,
        mode="eval",
        eval_part = eval_part),
    # test
    test_samples_per_gpu=1,
    test_workers_per_gpu=4,
    test=dict(
        type=test_dataset_type,
        lq_folder= "/home/megstudio/workspace/test/test2",
        num_input_frames = frames,
        pipeline=test_pipeline,
        scale=scale,
        mode="test")
)

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=8/16/10 * 1e-4, betas=(0.9, 0.999)))

# learning policy
total_epochs = 100 // repeat_times

# hooks
lr_config = dict(policy='Step', step=[total_epochs // 10], gamma=0.7)
checkpoint_config = dict(interval=total_epochs // 50)
log_config = dict(
    interval=300,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='VisualDLLoggerHook')
    ])
visual_config = None
evaluation = dict(interval=20000, save_image=True)

# runtime settings
work_dir = f'./workdirs/{exp_name}'
load_from = '/home/megstudio/workspace/checkpoints/mucan_v2/epoch_66'
resume_from = None
resume_optim = True
workflow = 'test'

# logger
log_level = 'INFO'

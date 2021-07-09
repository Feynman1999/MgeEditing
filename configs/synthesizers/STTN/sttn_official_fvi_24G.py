"""
ECCV2020 STTN
Learning Joint Spatial-Temporal Transformations for Video Inpainting

training:
select five frames Continuously or randomly(ordered)

test:
15 frames for 5 frames

one stage model, have G and D
G: end to end conv transformer  (L1 loss + GAN loss)
D: 3Dconv spatial-temporal GAN  (hingle loss)
"""
load_from = "/data/home/songtt/chenyuxiang/MBRVSR/workdirs/sttn_official_fvi_24G/20210524_204828/checkpoints/epoch_700"
dataroot = "/data/home/songtt/chenyuxiang/datasets/FVI/Train/JPEGImages"
exp_name = 'sttn_official_fvi_with_mask'
input_h = 240
input_w = 432
input_nums = 5
lr = 1 * 1e-4 * 0.1
loss_weight = {
    "hole_weight": 1,
    "valid_weight": 1,
    "adversarial_weight": 0.01,
}

# you can custom values before, for the following params do not change if you are new to this project
###########################################################################################

# model settings
model = dict(
    type='STTN_synthesizer',
    generator=dict(
        type='STTN',
        channels = 64,
        layers = 8, # 3
        heads = 4),
    discriminator = dict(
        type='Discriminator',
        use_sigmoid = False,
        use_sn = True
    ),
    pixel_loss = dict(type='L1Loss'),
    adv_loss = dict(type='AdversarialLoss', losstype='hinge'),
    loss_weight = loss_weight
)

# model training and testing settings
train_cfg = None
eval_cfg = dict(metrics=['PSNR'])
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

# dataset settings
train_dataset_type = 'VosInpaintingDataset'
eval_dataset_type = 'VosInpaintingDataset'

train_pipeline = [
    dict(type='GenerateRandomMasks', h=input_h, w=input_w),
    dict(type='GenerateContinuousOrDiscontinuousIndices', continuous_probability = 0.5, inputnums = input_nums), # should have frames_path
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='frames',
        flag='unchanged',
        make_bin=False),
    dict(type='Resize', keys=['frames'], size=(input_h, input_w), interpolation='area'),
    dict(type='RescaleToZeroOne', keys=['frames', 'masks']),
    dict(type='Normalize', keys=['frames'], to_rgb=True, **img_norm_cfg),
    dict(type='FramesToTensor', keys=['frames', 'masks']),
    dict(type='Collect', keys=['frames', 'masks'])
]

eval_pipeline = [
    dict(type='GenerateRandomMasks', h=input_h, w=input_w),
    dict(type='GenerateContinuousOrDiscontinuousIndices', continuous_probability = 1), # should have frames_path
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='frames',
        flag='unchanged',
        make_bin=False),
    dict(type='Resize', keys=['frames'], size=(input_h, input_w), interpolation='area'),
    dict(type='RescaleToZeroOne', keys=['frames', 'masks']),
    dict(type='Normalize', keys=['frames'], to_rgb=True, **img_norm_cfg),
    dict(type='FramesToTensor', keys=['frames', 'masks']),
    dict(type='Collect', keys=['frames', 'masks'])
]

repeat_times = 1
eval_part = None
data = dict(
    # train
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=repeat_times,
        dataset=dict(
            type=train_dataset_type,
            folder= dataroot,
            pipeline=train_pipeline,
            eval_part = eval_part)),
    # eval
    eval_samples_per_gpu=1,
    eval_workers_per_gpu=4,
    eval=dict(
        type=eval_dataset_type,
        folder= dataroot,
        pipeline=eval_pipeline,
        mode="eval",
        eval_part = eval_part)
)

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=lr, betas=(0.0, 0.99)),
                    discriminator = dict(type='Adam', lr=lr, betas=(0.0, 0.99)))

# learning policy
total_epochs = 1500 // repeat_times

# hooks
checkpoint_config = dict(interval=50)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', average_length=300),
        # dict(type='VisualDLLoggerHook')
    ])
evaluation = dict(interval=60000000000, save_image=False, multi_process=False, ensemble=False)

# runtime settings
work_dir = f'./workdirs/{exp_name}'
resume_from = None
resume_optim = True
workflow = 'train'

# logger
log_level = 'INFO'

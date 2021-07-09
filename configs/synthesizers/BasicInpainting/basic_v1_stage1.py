"""
Bi-RNN for video inpainting

training:
select 10 frames Continuously or randomly(ordered)

test:
100 frames

one stage model, have G and D
G: end to end Bi-RNN  (L1 loss + GAN loss)
D: 3Dconv spatial-temporal GAN  (hingle loss)
"""
load_from = None
path2spynet = "./workdirs/spynet/spynet-sintel-final.mge"
dataroot = "/data/home/songtt/chenyuxiang/datasets/mgtv"
exp_name = 'BasicInpainting'
flow_lr_mult = 0
input_h = 576 // 2
input_w = 1024 // 2
input_nums = 15
lr = 1 * 1e-4
loss_weight = {
    "hole_weight": 1,
    "valid_weight": 1,
    "adversarial_weight": 0.02,
}

# you can custom values before, for the following params do not change if you are new to this project
###########################################################################################

# model settings
model = dict(
    type='basicinpaint_synthesizer',
    generator=dict(
        type='BASIC_Inpaint',
        reconstruction_blocks = 4,
        blocknums = 6,
        flownet_layers = 4,
        pretrained_optical_flow_path = path2spynet),
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
    dict(type='RandomCrop', keys=['frames'], patch_size=(input_h, input_w)),
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
    dict(type='RescaleToZeroOne', keys=['frames', 'masks']),
    dict(type='Normalize', keys=['frames'], to_rgb=True, **img_norm_cfg),
    dict(type='FramesToTensor', keys=['frames', 'masks']),
    dict(type='Collect', keys=['frames', 'masks'])
]

repeat_times = 1
data = dict(
    # train
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=repeat_times,
        dataset=dict(
            type=train_dataset_type,
            folder= dataroot + "/train/train_all_pngs",
            pipeline=train_pipeline)),
    # eval
    eval_samples_per_gpu=1,
    eval_workers_per_gpu=4,
    eval=dict(
        type=eval_dataset_type,
        folder= dataroot + "/train/train_all_pngs",
        pipeline=eval_pipeline,
        mode="eval")
)

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=lr, betas=(0.9, 0.999), paramwise_cfg=dict(custom_keys={
                                                    'flownet': dict(lr_mult=flow_lr_mult)})),
                    discriminator = dict(type='Adam', lr=lr, betas=(0.9, 0.999)))

# learning policy
total_epochs = 1000 // repeat_times

# hooks
checkpoint_config = dict(interval=30)
log_config = dict(
    interval=10,
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

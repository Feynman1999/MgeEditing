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
load_from = "./workdirs/BasicInpainting/20210624_164450/checkpoints/epoch_240"
path2spynet = "./workdirs/spynet/spynet-sintel-final.mge"
dataroot = "/data/home/songtt/chenyuxiang/datasets/mgtv/test/test_a"
exp_name = 'BasicInpainting_test'
flow_lr_mult = 0
input_h = 576
input_w = 1024
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
test_dataset_type = 'MGTVInpaintingDataset'

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='img',
        flag='color'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='mask',
        flag='grayscale'),
    dict(type='RescaleToZeroOne', keys=['img', 'mask']),
    dict(type='Normalize', keys=['img'], to_rgb=True, **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img', 'mask']),
    dict(type='Collect', keys=['img', 'mask', 'index', 'max_frame_num'])
]

repeat_times = 1
data = dict(
    test_samples_per_gpu=1,
    test_workers_per_gpu=5,
    test=dict(
        type=test_dataset_type,
        folder= dataroot,
        pipeline=test_pipeline,
        mode="test")
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
workflow = 'test'

# logger
log_level = 'INFO'

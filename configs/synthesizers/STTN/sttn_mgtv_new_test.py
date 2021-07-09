load_from = "./workdirs/sttn_mgtv_with_mask/20210612_233333/checkpoints/epoch_1000"
dataroot = "/data/home/songtt/chenyuxiang/datasets/mgtv/test/test_a"
exp_name = 'sttn_mgtv_with_mask_test'
input_h = 576
input_w = 1024
lr = 1 * 1e-4
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
        type='STTN_MGTV',
        channels = 64,
        layers = 6, # 3
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
optimizers = dict(generator=dict(type='Adam', lr=lr, betas=(0.0, 0.99)),
                    discriminator = dict(type='Adam', lr=lr, betas=(0.0, 0.99)))

# learning policy
total_epochs = 1000 // repeat_times

# hooks
checkpoint_config = dict(interval=50)
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

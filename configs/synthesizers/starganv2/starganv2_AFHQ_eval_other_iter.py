exp_name = 'starganv2_AFHQ_iter'

# model settings
model = dict(
    type='STARGANV2',
    generator=dict(
        type='STARGANV2_G',
        img_size=256,
        style_dim=64,
        max_conv_dim=512,
        w_hpf=1),
    compon_M=dict(
        type='MappingNetwork',
        latent_dim=16,
        style_dim=64,
        num_domains=2),
    compon_S=dict(
        type='StyleEncoder',
        img_size=256, 
        style_dim=64, 
        num_domains=2, 
        max_conv_dim=512),
    compon_D=dict(
        type='Discriminator',
        img_size=256, 
        num_domains=2, 
        max_conv_dim=512),
    lambda_reg = 1.0,
    lambda_sty=1.0, 
    lambda_ds=2.0, 
    lambda_cyc=1.0
)

# model training and testing settings
train_cfg = None
eval_cfg = dict(metrics=['FID', 'LPIPS'])

img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# dataset settings
train_dataset_type = 'STARGANV2'
eval_dataset_type = 'STARGANV2'
img_keys = ['x', 'x_ref', 'x_ref2']
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='x',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='x_ref',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='x_ref2',
        flag='unchanged'),
    dict(type='RandomResizedCrop', keys=img_keys, output_size=256, scale=(0.8, 1.0), ratio=(0.9, 1.1), do_prob=0.5),
    dict(type='Resize', keys=img_keys, size=(256, 256)),
    dict(type='RescaleToZeroOne', keys=img_keys),
    dict(type='Normalize', keys=img_keys, to_rgb=True, **img_norm_cfg),
    dict(type='Flip', keys=img_keys, flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=img_keys, flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=img_keys, transpose_ratio=0.5),
    dict(type='ImageToTensor', keys=img_keys),
    dict(type='Collect', keys=['x','y','x_ref','y_ref','x_ref2','z_trg', 'z_trg2'])
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='img',
        flag='unchanged'),
    dict(type='Resize', keys=['img'], size=(256, 256)),
    dict(type='RescaleToZeroOne', keys=['img']),
    dict(type='Normalize', keys=['img'], to_rgb=True, **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

# imagenet_normalize or other
# use other below
eval_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='img',
        flag='unchanged'),
    dict(type='Resize', keys=['img'], size=(256, 256)),
    dict(type='RescaleToZeroOne', keys=['img']),
    dict(type='Normalize', keys=['img'], to_rgb=True, **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'domain_id'])
]

dataroot = "/opt/data/private/datasets"
repeat_times = 1000
data = dict(
    # train
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=repeat_times,
        dataset=dict(
            type=train_dataset_type,
            path=dataroot + "/afhq/train",
            pipeline=train_pipeline,
            num_domains=3)),
    # eval
    val_samples_per_gpu=16,
    val_workers_per_gpu=4,
    val=dict(
        type=eval_dataset_type,
        path=dataroot + "/afhq/val",
        pipeline=eval_pipeline,
        num_domains=3,
        test_mode=True),
    # test (sample)
    test=dict(
        type=eval_dataset_type,
        path=dataroot + "/afhq/val",
        pipeline=test_pipeline,
        num_domains=3,
        test_mode=True)
)

# optimizers
optimizers = dict(
        generator=dict(type='AdamOptimizer', learning_rate=1e-4, beta1=0, beta2=0.99, weight_decay=1e-4),
        compon_M=dict(type='AdamOptimizer', learning_rate=1e-6, beta1=0, beta2=0.99, weight_decay=1e-4),
        compon_S=dict(type='AdamOptimizer', learning_rate=1e-4, beta1=0, beta2=0.99, weight_decay=1e-4),
        compon_D=dict(type='AdamOptimizer', learning_rate=1e-4, beta1=0, beta2=0.99, weight_decay=1e-4)
)

# runtime settings
total_iters = 100000

# hooks
lr_config = dict(policy='Step', step=[total_iters // 10], gamma=0.7)
checkpoint_config = dict(interval=total_iters/20, by_epoch=False)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit'))
    ])
visual_config = None
evaluation = dict(interval=total_iters // 100, save_image=False)

# runtime settings
work_dir = f'./workdirs/{exp_name}'
load_from = None
resume_from = None
resume_optim = True
workflow = [('train', 1)]

# logger
log_level = 'INFO'



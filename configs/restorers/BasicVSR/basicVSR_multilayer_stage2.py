load_from = "./workdirs/basicVSR_multi_layer_stage1/20210411_004249/checkpoints/epoch_6"
path2spynet = "./workdirs/spynet/spynet-sintel-final.mge"
dataroot = "/work_base/datasets/REDS/train"
exp_name = 'basicVSR_multi_layer_stage2'
is_shift = False

# you can custom values before, for the following params do not change if you are new to this project
###########################################################################################

scale = 4

# model settings
model = dict(
    type='MultilayerBidirectionalRestorer',
    generator=dict(
        type='BasicVSR_MULTI_LAYER',
        in_channels=3,
        out_channels=3,
        hidden_channels = 36,
        init_nums = 4, # 4
        blocknums = 4,
        reconstruction_blocks = 8,
        upscale_factor = scale,
        pretrained_optical_flow_path = path2spynet,
        flownet_layers = 4,
        blocktype = "resblock",
        RNN_layers = 4,
        is_shift = is_shift),
    pixel_loss=dict(type='CharbonnierLoss'),
    Fidelity_loss = None
)

# model training and testing settings
train_cfg = None
eval_cfg = dict(metrics=['PSNR'], crop_border=0, multi_pad = 1, gap = 1)
img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1])

# dataset settings
train_dataset_type = 'SRManyToManyDataset'
eval_dataset_type = 'SRManyToManyDataset'  # 统一都用这个dataset
test_dataset_type = 'SRManyToManyDataset'

train_pipeline = [
    dict(type='GenerateFrameIndices', interval_list=[1], many2many = True, index_start = 0, name_padding = True, load_flow = False),
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
    dict(type='PairedRandomCrop', gt_patch_size=[80 * 4, 80 * 4], crop_flow=False),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='Normalize', keys=['lq', 'gt'], to_rgb=True, **img_norm_cfg),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt', 'lq_path', 'gt_path'])
]

eval_pipeline = [
    dict(type='GenerateFrameIndiceswithPadding', padding="reflection", many2many = False, index_start = 0, name_padding = True),
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
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt', 'num_input_frames', 'LRkey', 'lq_path'])
]

repeat_times = 1
eval_part = ('000', '011', '015', '020')
data = dict(
    # train
    samples_per_gpu=3,
    workers_per_gpu=9,
    train=dict(
        type='RepeatDataset',
        times=repeat_times,
        dataset=dict(
            type=train_dataset_type,
            lq_folder= dataroot + "/train_sharp_bicubic/X4",
            gt_folder= dataroot + "/train_sharp",
            num_input_frames=19,
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
optimizers = dict(generator=dict(type='Adam', lr=1.5 * 1e-4, betas=(0.9, 0.999),
                                paramwise_cfg=dict(custom_keys={
                                                    'flownet': dict(lr_mult=0.1)})))

# learning policy
total_epochs = 400 // repeat_times

# hooks
lr_config = dict(policy='Step', step=[total_epochs // 10], gamma=0.7)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=3,
    hooks=[
        dict(type='TextLoggerHook', average_length=2000),
        # dict(type='VisualDLLoggerHook')
    ])
evaluation = dict(interval=1200, save_image=False, multi_process=False, ensemble=False)

# runtime settings
work_dir = f'./workdirs/{exp_name}'
resume_from = None
resume_optim = True
workflow = 'train'

# logger
log_level = 'INFO'

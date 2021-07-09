"""
    Multilayer Bidirectional Recurrent Network for Video Super Resolution
"""
load_from = None
path2spynet = "./workdirs/spynet/spynet-sintel-final.mge"
dataroot = "/work_base/datasets/REDS/train"
exp_name = 'MBRVSR_multilayer_stage1'
blocktype = "resblock"
samples_per_gpu = 7

num_input_frames = 15
hidden_channels = 32
blocknums = 4
RNN_layers = 4
reconstruction_blocks = 4
flownet_layers = 4
flow_lr_mult = 0

eval_gap = 950 * 2 # iter
log_gap = 10 # iter

# you can custom values before, for the following params do not change if you are new to this project
###########################################################################################
use_highway = False
use_flow_mask = False
use_gap = False
if use_flow_mask:
    assert use_highway, "if use flow mask, please use highway"
if use_gap:
    assert use_highway and use_flow_mask, "if use gap, please use highway and flow mask"

learning_rate_per_batch = 1e-4
crop_size = 64
scale = 4

# model settings
model = dict(
    type='MultilayerBidirectionalRestorer',
    generator=dict(
        type='BasicVSR_MULTI_LAYER',
        in_channels=3,
        out_channels=3,
        hidden_channels = hidden_channels,
        blocknums = blocknums,
        reconstruction_blocks = reconstruction_blocks,
        upscale_factor = scale,
        pretrained_optical_flow_path = path2spynet,
        flownet_layers = flownet_layers,
        blocktype = blocktype,
        RNN_layers = RNN_layers),
    pixel_loss=dict(type='CharbonnierLoss'),
    Fidelity_loss = None
)

# model training and testing settings
img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1])
train_cfg = dict(img_norm_cfg = img_norm_cfg, train_cfg = dict(use_highway = use_highway, use_flow_mask = use_flow_mask, use_gap = use_gap))
eval_cfg = dict(metrics=['PSNR'], crop_border=0, multi_pad = 1, gap = 1)

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
    dict(type='PairedRandomCrop', gt_patch_size=[crop_size * 4, crop_size * 4], crop_flow=False),
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
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=samples_per_gpu * 3,
    train=dict(
        type='RepeatDataset',
        times=repeat_times,
        dataset=dict(
            type=train_dataset_type,
            lq_folder= dataroot + "/train_sharp_bicubic/X4",
            gt_folder= dataroot + "/train_sharp",
            num_input_frames=num_input_frames,
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
optimizers = dict(generator=dict(type='Adam', lr=learning_rate_per_batch * samples_per_gpu, betas=(0.9, 0.999),
                                paramwise_cfg=dict(custom_keys={
                                                    'flownet': dict(lr_mult=flow_lr_mult)})))

# learning policy
total_epochs = 400 // repeat_times

# hooks
lr_config = dict(policy='Step', step=[total_epochs // 10], gamma=0.7)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=log_gap,
    hooks=[
        dict(type='TextLoggerHook', average_length=500),
    ])
evaluation = dict(interval=eval_gap, save_image=False, multi_process=False, ensemble=False)

# runtime settings
work_dir = f'./workdirs/{exp_name}'
resume_from = None
resume_optim = True
workflow = 'train'

# logger
log_level = 'INFO'

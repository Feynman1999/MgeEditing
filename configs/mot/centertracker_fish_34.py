"""
ECCV2020   Tracking Objects as Points

training:
select two frames with three type noise
pre frame & now frame
data we need:
input:
1. pre image
2. now image
3. pre hot map label (generator by bbox with noise)
label:
1. now bbox (for 4 losses)

test:
any frames

one stage model, same as centerNet with a little different
"""
load_from = None
dataroot = "/data/home/songtt/chenyuxiang/datasets/MOTFISH/preliminary"
exp_name = 'centertracker_fish'
input_h = 480
input_w = 480
loss_weight = {
    "hms": 1,
    "hw": 1,
    "motion": 1,
}
fp = 0.1 # 在周围随机再加一个框的概率 默认0.1
fn = 0.4 # 去除一个框的概率 默认0.4
# you can custom values before, for the following params do not change if you are new to this project
###########################################################################################

# model settings
model = dict(
    type='ONLINE_MOT',
    generator=dict(
        type='CenterTrack',
        inp_h = input_h,
        inp_w = input_w,
        channels = 32,
        head_channels = 64,
        num_layers = 34,
        num_classes = 1,
        backbone_imagenet_pretrain = False,
        all_pretrain = False,
        min_overlap = 0.3,
        fp = fp,
        fn = fn),
    loss_weight = loss_weight
)

# model training and testing settings
train_cfg = None
eval_cfg = dict(metrics=['MOTA', 'IDF1'])
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])

# dataset settings
train_dataset_type = 'MotFishDataset'
eval_dataset_type = 'MotFishDataset'

train_pipeline = [
    dict(type='GenerateFrameIndices_now_and_pre', interval = 3), # should have img1_path and img2_path and Annotations
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='img1',
        flag='color'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='img2',
        flag='color'),
    dict(type='ColorJitter', keys=['img1', 'img2'], brightness=0.2, contrast=0.4, saturation=0.0, hue=0.0),
    dict(type='RescaleToZeroOne', keys=['img1', 'img2']),
    dict(
        type='RandomCenterCropPadTwo',
        ratios=(0.8, 0.9, 1.0, 1.1, 1.2),
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=False),
    dict(type='Resize', keys=['img1', 'img2'], size=[input_h, input_w], interpolation="area"),
    dict(type='Normalize', keys=['img1', 'img2'], to_rgb=True, **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img1', 'img2']),
    # pad annotations
    dict(type='PadAnnotations', max_len = 40, keys = ['gt_bboxes1', 'gt_bboxes2', 'gt_labels1', 'gt_labels2']), # 会add _num表示真实Object数量
    dict(type='Collect', keys=['img1', 'img2', 'gt_bboxes1', 'gt_bboxes2', 'gt_labels1', 'gt_labels2', 'gt_bboxes1_num', 'gt_bboxes2_num'])
]

eval_pipeline = [
    dict(type='GenerateFrameIndices_now_and_pre', interval = 1), # should have img1_path and img2_path and Annotations
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='img1',
        flag='color'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='img2',
        flag='color'),
    dict(type='RescaleToZeroOne', keys=['img1', 'img2']),
    dict(type='Resize', keys=['img1', 'img2'], size=[input_h, input_w], interpolation="area"),
    dict(type='Normalize', keys=['img1', 'img2'], to_rgb=True, **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img1', 'img2']),
    # pad annotations
    dict(type='PadAnnotations', max_len = 40, keys = ['gt_bboxes1', 'gt_bboxes2', 'gt_labels1', 'gt_labels2']), # 会add _num表示真实Object数量
    dict(type='Collect', keys=['img1', 'img2', 'gt_bboxes1', 'gt_bboxes2', 'gt_labels1', 'gt_labels2', 'gt_bboxes1_num', 'gt_bboxes2_num'])
]

repeat_times = 1
eval_part = ('r1', ) # 1/5
data = dict(
    # train
    samples_per_gpu=12,
    workers_per_gpu=12,
    train=dict(
        type='RepeatDataset',
        times=repeat_times,
        dataset=dict(
            type=train_dataset_type,
            folder= dataroot + '/train',
            pipeline=train_pipeline,
            eval_part = eval_part)),
    # eval
    eval_samples_per_gpu=1,
    eval_workers_per_gpu=4,
    eval=dict(
        type=eval_dataset_type,
        folder= dataroot + '/train',
        pipeline=eval_pipeline,
        mode="eval",
        eval_part = eval_part)
)

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=1e-4, betas=(0.99, 0.99)))

# learning policy
total_epochs = 100 // repeat_times

# hooks
checkpoint_config = dict(interval=5)
log_config = dict(
    interval=1,
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

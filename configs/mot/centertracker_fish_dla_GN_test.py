load_from = './xxxx/epoch_80' # xxxx为你自己放置的路径，注意目录要精确到epoch_xxx，xxx是数字
dataroot = "xxxxx/preliminary/test" # xxxx为你的数据集的路径
exp_name = 'centertracker_fish_test' # 名字随便取
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
        backbone_type = "DLA_GN",
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
test_dataset_type = 'MotFishTestDataset'

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='img',
        flag='color'),
    dict(type='Add_contrast', keys=['img'], value = 0.9),
    dict(type='RescaleToZeroOne', keys=['img']),
    dict(type='Resize', keys=['img'], size=[input_h, input_w], interpolation="area"),
    dict(type='Normalize', keys=['img'], to_rgb=True, **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'scale_factor', 'index', 'clipname', 'gap', 'total_len'])
]

repeat_times = 1
data = dict(
    test_samples_per_gpu=1,
    test_workers_per_gpu=5,
    test=dict(
        type=test_dataset_type,
        folder= dataroot,
        pipeline=test_pipeline)
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
workflow = 'test'

# logger
log_level = 'INFO'

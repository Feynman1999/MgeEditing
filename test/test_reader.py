import numpy as np

import paddle.fluid as fluid
from edit.datasets.sr_folder_dataset import SRFolderDataset
from paddle.io import DataLoader
from paddle.incubate.hapi.distributed import DistributedBatchSampler
from edit.utils import var2img, imwrite
lq_folder = r"E:\git_repo\Code-Implementation-of-Super-Resolution-ZOO\datasets\DIV2K\train\A"
gt_folder = r"E:\git_repo\Code-Implementation-of-Super-Resolution-ZOO\datasets\DIV2K\train\B"


train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='PairedRandomCrop', gt_patch_size=960),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt']),
]


def main():
    BATCH_SIZE = 1
    place = fluid.CPUPlace()
    dataset_obj = SRFolderDataset(lq_folder=lq_folder, gt_folder=gt_folder, pipeline=train_pipeline, scale=4)

    with fluid.dygraph.guard(place):
        train_sampler = DistributedBatchSampler(
            dataset_obj,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True)
        train_loader = DataLoader(
            dataset_obj,
            batch_sampler=train_sampler,
            places=place,
            num_workers=4,
            return_list=True)

        for batch_id, data in enumerate(train_loader):
            imwrite(var2img(data[1]), "./test/niu.png")
            break


if __name__ == "__main__":
    main()


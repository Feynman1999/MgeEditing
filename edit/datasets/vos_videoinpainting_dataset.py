"""
    for video inpainting dataset, e.g. FVI and VOS2018
"""
import os
from .base_videoinpainting_dataset import BaseVideoInpaintingDataset
from .registry import DATASETS
from edit.utils import scandir, mkdir_or_exist, is_tuple_of, is_list_of

IMG_EXTENSIONS = ('.png', '.jpg')

@DATASETS.register_module()
class VosInpaintingDataset(BaseVideoInpaintingDataset):
    def __init__(self,
                 folder,
                 pipeline,
                 mode = "train"):
        super(VosInpaintingDataset, self).__init__(pipeline, mode)
        self.folder = str(folder)
        self.data_infos = self.load_annotations()
        self.logger.info("VosInpaintingDataset dataset load ok,   mode: {}   len:{}".format(self.mode, len(self.data_infos)))

    def load_annotations(self):
        # get keys
        keys = os.listdir(self.folder)
        keys = [ v for v in keys if "." not in v]
        keys = sorted(keys)  # 0f2ab8b1ff for FVI  video_xxxx for mgtv
        
        self.frame_num = dict()
        for key in keys:
            self.frame_num[key] = len(list(scandir(os.path.join(self.folder, key), suffix=IMG_EXTENSIONS, recursive=False)))
            
        data_infos = []
        for key in keys:
            if self.mode == "train":
                data_infos.append(
                    dict(
                        path = self.folder,
                        key = key,
                        max_frame_num = self.frame_num[key]
                    )
                )
            elif self.mode == "eval":
                data_infos.append(
                    dict(
                        path = self.folder,
                        key = key,
                        max_frame_num = self.frame_num[key]
                    )
                )
            elif self.mode == "test":
                data_infos.append(
                    dict(
                        path = self.folder,
                        key = key,
                        max_frame_num = self.frame_num[key]
                    )
                )
            else:
                raise NotImplementedError("")
        return data_infos


@DATASETS.register_module()
class MGTVInpaintingDataset(BaseVideoInpaintingDataset):
    def __init__(self,
                 folder,
                 pipeline,
                 imgs_dir = "frames_corr",
                 masks_dir = "masks",
                 bbox_file = "minbbox.txt",
                 mode = "test"):
        super(MGTVInpaintingDataset, self).__init__(pipeline, mode)
        assert mode in ("eval", "test")
        self.folder = str(folder)
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.bbox_file = bbox_file
        self.data_infos = self.load_annotations()
        self.logger.info("MGTVInpaintingDataset dataset load ok,   mode: {}   len:{}".format(self.mode, len(self.data_infos)))

    def load_annotations(self):
        # get clips
        clips = os.listdir(self.folder)
        clips = [ v for v in clips if "." not in v] # 确保clip文件夹
        clips = sorted(clips)  # 0f2ab8b1ff for FVI  video_xxxx for mgtv
        self.logger.info("total {} video clip".format(len(clips)))
        # 对于每个clip，遍历里面的imgs和masks，给上路径即可，并加上所在clip的index(start from 0)，以及所在clip的长度
        data_infos = []
        for clip in clips:
            # 看当前一共有多少帧
            imgs = sorted(list(scandir(os.path.join(self.folder, clip, self.imgs_dir), suffix=IMG_EXTENSIONS, recursive=False)))
            masks = sorted(list(scandir(os.path.join(self.folder, clip, self.masks_dir), suffix=IMG_EXTENSIONS, recursive=False)))
            now_clip_framenum = len(imgs)
            assert now_clip_framenum == len(masks)
            # 按照顺序遍历img 和 mask
            for idx in range(now_clip_framenum):
                img_path = os.path.join(self.folder, clip, self.imgs_dir, imgs[idx])
                mask_path = os.path.join(self.folder, clip, self.masks_dir, masks[idx])
                if self.mode == "test":
                    data_infos.append(
                        dict(
                            img_path = img_path,
                            mask_path = mask_path,
                            max_frame_num = now_clip_framenum,
                            index = idx
                        )
                    )
                elif self.mode == "eval":
                    pass
                    raise NotImplementedError("")
                else:
                    raise NotImplementedError("")
        return data_infos

"""
    for mot fish dataset
    中国农业人工智能创新创业大赛
    https://studio.brainpp.com/competition/3
"""
import os
import numpy as np
from .base_mot_dataset import BaseMotDataset
from .registry import DATASETS
from edit.utils import scandir, mkdir_or_exist, is_tuple_of, is_list_of
from collections import defaultdict

IMG_EXTENSIONS = ('.PNG',)

def get_key_for_imgs(x):
    clip, _, name = x.split("/")
    assert _ == "img1"
    name, _ = os.path.splitext(name)
    return (clip, int(name))  # 按照clip升序sort，若属于同一clip，则按照frame number升序

@DATASETS.register_module()
class MotFishDataset(BaseMotDataset):
    def __init__(self,
                 folder,
                 pipeline,
                 mode = "train",
                 eval_part = None):
        super(MotFishDataset, self).__init__(pipeline, mode)
        self.folder = str(folder)
        self.eval_part = eval_part
        if eval_part is not None:
            assert is_tuple_of(eval_part, str)
        self.data_infos = self.load_annotations()
        self.logger.info("MotFishDataset dataset load ok,   mode: {}   len:{}".format(self.mode, len(self.data_infos)))

    def load_annotations(self):
        # get keys
        keys = list(scandir(self.folder, suffix=IMG_EXTENSIONS, recursive=True))
        keys = [ v for v in keys if len(v.split('/')) == 3]
        keys = sorted(keys, key=get_key_for_imgs)  # i3/img1/000001.PNG
        
        # do split for train and eval
        if self.eval_part is not None:
            if self.mode == "train":
                keys = [k for k in keys if k.split('/')[0] not in self.eval_part]
            elif self.mode == "eval":
                keys = [k for k in keys if k.split('/')[0] in self.eval_part]
            else:
                pass

        # 构建一个字典,key值是clip + imgname, value是当前帧的bboxes(s,4)和类别,id (s,2)
        label_dict = None
        if self.mode != "test":
            label_dict = defaultdict(list)
            for clip_name in os.listdir(self.folder):
                # 读取一个clip的所有gt
                label_txt = os.path.join(self.folder, clip_name, "gt", "gt.txt")
                with open(label_txt, "r") as f:
                    for line in f.readlines():
                        # key: clip_name + frame_id + bboxes/ id/ classes
                        values = line.split(",")[0:6]
                        values = [ int(float(v)+0.5) for v in values]
                        x,y,w,h = values[2:]
                        frame = str(values[0])
                        ID = values[1]
                        class_id = 0
                        # write to dict
                        label_dict[clip_name + "_" + frame + "_bboxes"].append(np.array([x,y,x+w,y+h])) # list of (4) numpy tl_x, tl_y, br_x, br_y
                        label_dict[clip_name + "_" + frame + "_labels"].append(np.array([class_id, ID])) # list of (2) numpy

        data_infos = []
        for key in keys:
            if self.mode == "train":
                data_infos.append(
                    dict(
                        folder = self.folder,
                        key = key,
                        label_dict = label_dict # 给clip和frame可以查到list of numpy，在pipeline中stack即可
                    )
                )
            elif self.mode == "eval":
                data_infos.append(
                    dict(
                        folder = self.folder,
                        key = key,
                        label_dict = label_dict # 给clip和frame可以查到list of numpy，在pipeline中stack即可
                    )
                )
            elif self.mode == "test":
                data_infos.append(
                    dict(
                        img_path = "",

                    )
                )
            else:
                raise NotImplementedError("")
        return data_infos

@DATASETS.register_module()
class MotFishTestDataset(BaseMotDataset):
    def __init__(self,
                 folder,
                 pipeline):
        super(MotFishTestDataset, self).__init__(pipeline, "test")
        self.folder = str(folder)
        self.data_infos = self.load_annotations()
        self.logger.info("MotFishTestDataset dataset load ok,   mode: {}   len:{}".format(self.mode, len(self.data_infos)))

    def add_infos(self, gap, infos, imgs, clipname):
        Len = 0
        for i in range(0, len(imgs), gap):
            Len += 1
        idx = 0
        for i in range(0, len(imgs), gap):
            infos.append(
                dict(
                    img_path = os.path.join(self.folder, clipname, "img1", imgs[i]),
                    total_len = Len,
                    index = idx,
                    clipname= clipname,
                    gap = gap
                )
            )
            idx+=1

    def load_annotations(self):
        data_infos = []
        for clipname in os.listdir(self.folder):
            imgs = sorted(list(scandir(os.path.join(self.folder, clipname, "img1"), suffix=IMG_EXTENSIONS, recursive=False)))
            self.add_infos(gap = 1, infos = data_infos, imgs = imgs, clipname = clipname)
            self.add_infos(gap = 5, infos = data_infos, imgs = imgs, clipname = clipname)
        return data_infos

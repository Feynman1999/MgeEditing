"""
    for many to many
"""
import os
from .base_sr_dataset import BaseVSRDataset
from .registry import DATASETS
from .utils import get_key_for_video_imgs
from edit.utils import scandir, mkdir_or_exist, is_tuple_of, is_list_of

IMG_EXTENSIONS = ('.png', )

@DATASETS.register_module()
class SRManyToManyDataset(BaseVSRDataset):
    """Many to Many dataset for video super resolution.

    The dataset loads several LQ (Low-Quality) frames and several GT
    (Ground-Truth) frames. Then it applies specified transforms and finally
    returns a list containing paired data (images, label).

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        num_input_frames (int): Window size for input frames.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        mode (str): "train", "test" or "eval"
    """

    def __init__(self,
                 lq_folder,
                 pipeline,
                 gt_folder = "",
                 num_input_frames = 7,
                 scale = 4,
                 mode = "train",
                 eval_part = None,
                 LR_symbol = "@"):  # impossible name default
        super(SRManyToManyDataset, self).__init__(pipeline, scale, mode)
        assert num_input_frames % 2 == 1, (
            f'num_input_frames should be odd numbers, '
            f'but received {num_input_frames }.')
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.num_input_frames = num_input_frames
        self.eval_part = eval_part
        if eval_part is not None:
            assert is_tuple_of(eval_part, str)
        self.LR_symbol = LR_symbol

        self.data_infos = self.load_annotations()
        
    def load_annotations(self):
        # get keys
        keys = list(scandir(self.lq_folder, suffix=IMG_EXTENSIONS, recursive=True))
        keys = [ v for v in keys if len(v.split('/')) == 2]
        keys = sorted(keys, key=get_key_for_video_imgs)  # 000/00000.png
        
        if self.lq_folder == self.gt_folder:
            # gt and lq in same dir, only select lq as keys
            keys = [key for key in keys if self.LR_symbol in key]

        # do split for train and eval   
        if self.eval_part is not None:
            if self.mode == "train":
                keys = [k for k in keys if k.split('/')[0] not in self.eval_part]
            elif self.mode == "eval":
                keys = [k for k in keys if k.split('/')[0] in self.eval_part]
            else:
                pass

        self.frame_num = dict()
        for key in keys:
            self.frame_num[key.split("/")[0]] = 0
        for key in keys:
            self.frame_num[key.split("/")[0]] += 1
        
        # for meg competition
        sence = dict()
        # sence['92'] = []
        # sence['92'].append(304)
        # sence['93'] = []
        # sence['93'].append(156)
        # sence['93'].append(449)
        # sence['93'].append(740)
        # sence['94'] = []
        # sence['94'].append(363)
        # sence['94'].append(375)
        # sence['95'] = []
        # sence['95'].append(196)
        # sence['95'].append(272)
        # sence['95'].append(347)
        # sence['96'] = []
        # sence['96'].append(120)
        # sence['97'] = []
        # sence['97'].append(53)
        # sence['99'] = []
        # sence['99'].append(181)

        data_infos = []
        is_first = 1
        now_deal = 0
        for key in keys:
            # do some checks, to make sure the key for LR and HR is same. 
            if self.mode in ("train", "eval"):
                gt_path = os.path.join(self.gt_folder, key.replace(self.LR_symbol, ""))
                assert os.path.exists(gt_path), "please make sure the key {} for LR and HR is same".format(key)

            if self.mode == "train":
                data_infos.append(
                    dict(
                        lq_path=self.lq_folder,
                        gt_path=self.gt_folder,
                        LRkey=key,
                        HRkey=key.replace(self.LR_symbol, ""),
                        max_frame_num=self.frame_num[key.split("/")[0]],
                        num_input_frames=self.num_input_frames
                    )
                )
            elif self.mode == "eval":
                data_infos.append(
                    dict(
                        lq_path = os.path.join(self.lq_folder, key),
                        gt_path = os.path.join(self.gt_folder, key.replace(self.LR_symbol, "")),
                        is_first = is_first
                    )
                )
            elif self.mode == "test":
                data_infos.append(
                    dict(
                        lq_path = os.path.join(self.lq_folder, key),
                        is_first = is_first
                    )
                )
            else:
                raise NotImplementedError("")

            # update is_first
            now_deal += 1
            if now_deal == self.frame_num[key.split("/")[0]]:
                is_first = 1
                now_deal = 0
            elif key.split("/")[0] in sence.keys() and now_deal in sence[key.split("/")[0]]:
                is_first = 1
            else:
                is_first = 0
        return data_infos

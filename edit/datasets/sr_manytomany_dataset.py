"""
    for many to many
"""
import os
from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS
from .utils import get_key_for_video_imgs
from edit.utils import scandir

IMG_EXTENSIONS = ('.png', )

@DATASETS.register_module()
class SRManyToManyDataset(BaseSRDataset):
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
                 mode = "train"):
        super(SRManyToManyDataset, self).__init__(pipeline, scale, mode)
        assert num_input_frames % 2 == 1, (
            f'num_input_frames should be odd numbers, '
            f'but received {num_input_frames }.')
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.num_input_frames = num_input_frames
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        # get keys
        keys = sorted(list(scandir(self.lq_folder, suffix=IMG_EXTENSIONS, recursive=True)),
                        key=get_key_for_video_imgs)  # 000/00000.png
        
        self.frame_num = dict()
        for key in keys:
            self.frame_num[key.split("/")[0]] = 0
        for key in keys:
            self.frame_num[key.split("/")[0]] += 1

        data_infos = []
        for key in keys:
            # do some checks, to make sure the key for LR and HR is same. 
            if self.mode in ("train", "eval"):
                gt_path = os.path.join(self.gt_folder, key)
                assert os.path.exists(gt_path), "please make sure the key for LR and HR is same"
            
            if self.mode == "train":
                data_infos.append(
                    dict(
                        lq_path=self.lq_folder,
                        gt_path=self.gt_folder,
                        key=key,
                        max_frame_num=self.frame_num[key.split("/")[0]],
                        num_input_frames=self.num_input_frames))
            elif self.mode == "eval":
                data_infos.append(
                    dict(
                        lq_path = os.path.join(self.lq_folder, key),
                        gt_path = os.path.join(self.gt_folder, key)
                    )
                )
            elif self.mode == "test":
                data_infos.append(
                    dict(
                        lq_path = os.path.join(self.lq_folder, key)
                    )
                )
            else:
                raise NotImplementedError("")
        return data_infos

import random
import math
import numpy as np
from ..registry import PIPELINES
from edit.utils import imresize


@PIPELINES.register_module()
class Random_Crop_Opt_Sar(object):
    def __init__(self, keys, size, have_seed = False, Contrast=False):
        self.keys = keys
        self.size = size # 500, 320
        self.have_seed = have_seed
        self.Contrast = Contrast
        if self.Contrast:
            assert have_seed == False

    def get_optical_h_w(self, sar_h, sar_w):
        up = 800 - self.size[0]
        optical_h = random.randint(max(sar_h - (self.size[0]-self.size[1]), 0), min(sar_h, up))
        optical_w = random.randint(max(sar_w - (self.size[0]-self.size[1]), 0), min(sar_w, up))
        return optical_h, optical_w

    def __call__(self, results):
        if self.have_seed:  # 用于测试时
            random.seed(np.sum(results['sar']))

        gap = 512 - self.size[1]
        sar_h = random.randint(0, gap) # 随机两个数 去裁剪sar
        sar_w = random.randint(0, gap)
        # 获得sar图像
        results['sar'] = results['sar'][sar_h:sar_h+self.size[1], sar_w:sar_w+self.size[1], :]  # h,w,1
        # 所以我们可以得到裁剪出的sar图在800中的左上角
        sar_h = results['bbox'][0] + sar_h
        sar_w = results['bbox'][1] + sar_w
        
        if self.Contrast: # 随机三个用于训练
            sar = results['sar']
            optical = results['opt']
            results['sar'] = []
            results['opt'] = []
            results['bbox'] = []
            for _ in range(3):
                results['sar'].append(sar.copy())
                opt = optical.copy()
                optical_h, optical_w = self.get_optical_h_w(sar_h, sar_w)
                results['opt'].append(opt[optical_h:optical_h+self.size[0], optical_w:optical_w+self.size[0], :])
                results['bbox'].append(np.array([sar_h - optical_h, 
                                                 sar_w - optical_w, 
                                                 sar_h - optical_h + self.size[1] - 1, 
                                                 sar_w - optical_w + self.size[1] - 1]).astype(np.float32))
        else:
            optical_h, optical_w = self.get_optical_h_w(sar_h, sar_w)
            results['opt'] = results['opt'][optical_h:optical_h+self.size[0], optical_w:optical_w+self.size[0], :]  # h,w,1

            # 更改bbox
            results['bbox'][0] = sar_h - optical_h
            results['bbox'][1] = sar_w - optical_w
            results['bbox'][2] = results['bbox'][0] + self.size[1] - 1
            results['bbox'][3] = results['bbox'][1] + self.size[1] - 1
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(keys={self.keys})')
        return repr_str


@PIPELINES.register_module()
class PairedRandomCrop(object):
    """Paried random crop.

    It crops a pair of lq and gt images with corresponding locations.
    It also supports accepting lq list and gt list.
    Required keys are "scale", "lq", and "gt",
    added or modified keys are "lq" and "gt".

    Args:
        gt_patch_size (int): cropped gt patch size.
    """

    def __init__(self, gt_patch_size):
        self.gt_patch_size = gt_patch_size

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        scale = results['scale']
        assert self.gt_patch_size % scale == 0
        lq_patch_size = self.gt_patch_size // scale

        lq_is_list = isinstance(results['lq'], list)
        if not lq_is_list:
            results['lq'] = [results['lq']]
        gt_is_list = isinstance(results['gt'], list)
        if not gt_is_list:
            results['gt'] = [results['gt']]

        h_lq, w_lq, _ = results['lq'][0].shape
        h_gt, w_gt, _ = results['gt'][0].shape

        if h_gt != h_lq * scale or w_gt != w_lq * scale:
            raise RuntimeError("HR's size is not {}X times to LR's size".format(scale))
            # do resize, resize gt to lq * scale
            # results['gt'] = [
            #     imresize(v, (w_lq * scale, h_lq * scale))
            #     for v in results['gt']
            # ]
            
        if h_lq < lq_patch_size or w_lq < lq_patch_size:
            raise ValueError(
                f'LQ ({h_lq}, {w_lq}) is smaller than patch size ',
                f'({lq_patch_size}, {lq_patch_size}). Please check '
                f'{results["lq_path"][0]} and {results["gt_path"][0]}.')

        # randomly choose top and left coordinates for lq patch
        top = random.randint(0, h_lq - lq_patch_size)
        left = random.randint(0, w_lq - lq_patch_size)
        # crop lq patch
        results['lq'] = [
            v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
            for v in results['lq']
        ]
        # crop corresponding gt patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        results['gt'] = [
            v[top_gt:top_gt + self.gt_patch_size,
              left_gt:left_gt + self.gt_patch_size, ...] for v in results['gt']
        ]

        if not lq_is_list:
            results['lq'] = results['lq'][0]
        if not gt_is_list:
            results['gt'] = results['gt'][0]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(gt_patch_size={self.gt_patch_size})'
        return repr_str

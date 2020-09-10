import numpy as np
import os.path as osp
from ..registry import PIPELINES
from edit.utils import imflip_


@PIPELINES.register_module()
class RandomTransposeHW(object):
    """Randomly transpose images in H and W dimensions with a probability.

    (TransposeHW = horizontal flip + anti-clockwise rotatation by 90 degrees)
    When used with horizontal/vertical flips, it serves as a way of rotation
    augmentation.
    It also supports randomly transposing a list of images.

    Required keys are the keys in attributes "keys", added or modified keys are
    "transpose" and the keys in attributes "keys".

    Args:
        keys (list[str]): The images to be transposed.
        transpose_ratio (float): The propability to transpose the images.
    """

    def __init__(self, keys, transpose_ratio=0.5):
        self.keys = keys
        self.transpose_ratio = transpose_ratio

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        transpose = np.random.random() < self.transpose_ratio

        if transpose:
            for key in self.keys:
                if isinstance(results[key], list):
                    results[key] = [v.transpose(1, 0, 2) for v in results[key]]
                else:
                    results[key] = results[key].transpose(1, 0, 2)

        results['transpose'] = transpose

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(keys={self.keys}, transpose_ratio={self.transpose_ratio})')
        return repr_str


@PIPELINES.register_module()
class Flip(object):
    """Flip the input data with a probability.

    Reverse the order of elements in the given data with a specific direction.
    The shape of the data is preserved, but the elements are reordered.
    Required keys are the keys in attributes "keys", added or modified keys are
    "flip", "flip_direction" and the keys in attributes "keys".
    It also supports flipping a list of images with the same flip.

    Args:
        keys (list[str]): The images to be flipped.
        flip_ratio (float): The propability to flip the images.
        direction (str): Flip images horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self, keys, flip_ratio=0.5, direction='horizontal'):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported.'
                             f'Currently support ones are {self._directions}')
        self.keys = keys
        self.flip_ratio = flip_ratio
        self.direction = direction

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        flip = np.random.random() < self.flip_ratio

        if flip:
            for key in self.keys:
                if isinstance(results[key], list):
                    for v in results[key]:
                        imflip_(v, self.direction)
                else:
                    imflip_(results[key], self.direction)

        results['flip'] = flip
        results['flip_direction'] = self.direction

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, flip_ratio={self.flip_ratio}, '
                     f'direction={self.direction})')
        return repr_str


@PIPELINES.register_module()
class GenerateFrameIndiceswithPadding(object):
    """Generate frame index with padding for Many to One VSR when test and eval.

    Required keys: lq_path, gt_path, key, num_input_frames, max_frame_num
    Added or modified keys: lq_path, gt_path

    Args:
         padding (str): padding mode, one of
            'replicate' | 'reflection' | 'reflection_circle' | 'circle'.

            Examples: current_idx = 0, num_input_frames = 5
            The generated frame indices under different padding mode:

                replicate: [0, 0, 0, 1, 2]
                reflection: [2, 1, 0, 1, 2]
                reflection_circle: [4, 3, 0, 1, 2]
                circle: [3, 4, 0, 1, 2]
    """

    def __init__(self, padding, index_start = 1, name_padding = False):
        if padding not in ('replicate', 'reflection', 'reflection_circle', 'circle'):
            raise ValueError(f'Wrong padding mode {padding}.'
                             'Should be "replicate", "reflection", '
                             '"reflection_circle",  "circle"')
        self.padding = padding
        self.index_start = index_start
        self.name_padding = name_padding

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        clip_name, frame_name = results['LRkey'].split('/') # 000/0000001.png
        if 'HRkey' in results.keys():
            clip_name_HR, _ = results['HRkey'].split('/')
        frame_name, ext_name = osp.splitext(frame_name)
        if self.name_padding:
            padding_length = len(frame_name)
        else:
            padding_length = 0
        current_idx = int(frame_name) - self.index_start  # start from 0
        max_frame_num = results['max_frame_num'] - 1  
        num_input_frames = results['num_input_frames']
        num_pad = num_input_frames // 2

        frame_list = []
        for i in range(current_idx - num_pad, current_idx + num_pad + 1):
            if i < 0:
                if self.padding == 'replicate':
                    pad_idx = 0
                elif self.padding == 'reflection':
                    pad_idx = -i
                elif self.padding == 'reflection_circle':
                    pad_idx = current_idx + num_pad - i
                else:
                    pad_idx = num_input_frames + i
            elif i > max_frame_num:
                if self.padding == 'replicate':
                    pad_idx = max_frame_num
                elif self.padding == 'reflection':
                    pad_idx = max_frame_num * 2 - i
                elif self.padding == 'reflection_circle':
                    pad_idx = (current_idx - num_pad) - (i - max_frame_num)
                else:
                    pad_idx = i - num_input_frames
            else:
                pad_idx = i
            frame_list.append(pad_idx)

        lq_path_root = results['lq_path']
        gt_path_root = results['gt_path']
        
        lq_paths = [
            osp.join(lq_path_root, clip_name,  str(idx + self.index_start).zfill(padding_length) +ext_name)
            for idx in frame_list
        ]
        gt_paths = [osp.join(gt_path_root, clip_name_HR, frame_name + ext_name)]
        results['lq_path'] = lq_paths
        results['gt_path'] = gt_paths

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f"(padding='{self.padding}')"
        return repr_str


@PIPELINES.register_module()
class GenerateFrameIndices(object):
    """Generate frame index for many to many or many to one datasets. It also performs
    temporal augmention with random interval.

    Required keys: lq_path, gt_path, key, num_input_frames, max_frame_num
    Added or modified keys:  lq_path, gt_path, interval

    Args:
        interval_list (list[int]): Interval list for temporal augmentation.
            It will randomly pick an interval from interval_list and sample
            frame index with the interval.
    """

    def __init__(self, interval_list, many2many = False, index_start = 1, name_padding = False):
        self.interval_list = interval_list
        self.many2many = many2many
        self.index_start = index_start
        self.name_padding = name_padding

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        clip_name, frame_name = results['LRkey'].split('/')  # key example: 000_down/00000000.png
        clip_name_HR, _ = results['HRkey'].split('/')  # key example: 000/00000000.png
        
        frame_name, ext_name = osp.splitext(frame_name)
        if self.name_padding:  # 由int恢复str时使用, int 方便下标计算
            padding_length = len(frame_name)
        else:
            padding_length = 0
        center_frame_idx = int(frame_name)
        num_half_frames = results['num_input_frames'] // 2

        interval = np.random.choice(self.interval_list)
        # ensure not exceeding the borders
        start_frame_idx = center_frame_idx - num_half_frames * interval
        end_frame_idx = center_frame_idx + num_half_frames * interval
        start = self.index_start
        end = start + results['max_frame_num']
        while (start_frame_idx < start) or (end_frame_idx >= end):
            center_frame_idx = np.random.randint(start, end)
            start_frame_idx = center_frame_idx - num_half_frames * interval
            end_frame_idx = center_frame_idx + num_half_frames * interval
        frame_name = str(center_frame_idx).zfill(padding_length) # 由于center_frame_idx可能改变，所以重新zfill
        neighbor_list = list(
            range(center_frame_idx - num_half_frames * interval,
                  center_frame_idx + num_half_frames * interval + 1, interval))

        lq_path_root = results['lq_path']
        gt_path_root = results['gt_path']
        lq_path = [
            osp.join(lq_path_root, clip_name, str(v).zfill(padding_length) + ext_name)
            for v in neighbor_list
        ]
        if self.many2many:
            gt_path = [
                osp.join(gt_path_root, clip_name_HR, str(v).zfill(padding_length) + ext_name)
                for v in neighbor_list
            ]
        else:
            gt_path = [osp.join(gt_path_root, clip_name_HR, frame_name + ext_name)]
        results['lq_path'] = lq_path
        results['gt_path'] = gt_path
        results['interval'] = interval
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f', interval_list={self.interval_list}'
        return repr_str


@PIPELINES.register_module()
class TemporalReverse(object):
    """Reverse frame lists for temporal augmentation.

    Required keys are the keys in attributes "lq" and "gt",
    added or modified keys are "lq", "gt" and "reverse".

    Args:
        keys (list[str]): The frame lists to be reversed.
        reverse_ratio (float): The propability to reverse the frame lists.
            Default: 0.5.
    """

    def __init__(self, keys, reverse_ratio=0.5):
        self.keys = keys
        self.reverse_ratio = reverse_ratio

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        reverse = np.random.random() < self.reverse_ratio

        if reverse:
            for key in self.keys:
                results[key].reverse()

        results['reverse'] = reverse

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}, reverse_ratio={self.reverse_ratio})'
        return repr_str
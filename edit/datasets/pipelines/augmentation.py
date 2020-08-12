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
class GenerateFrameIndices(object):
    """Generate frame index for REDS datasets. It also performs
    temporal augmention with random interval.

    Required keys: lq_path, gt_path, key, num_input_frames
    Added or modified keys:  lq_path, gt_path, interval, reverse

    Args:
        interval_list (list[int]): Interval list for temporal augmentation.
            It will randomly pick an interval from interval_list and sample
            frame index with the interval.
        frames_per_clip(int): Number of frames per clips. Default: 100 for
            REDS dataset.
    """

    def __init__(self, interval_list, frames_per_clip=100):
        self.interval_list = interval_list
        self.frames_per_clip = frames_per_clip

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        clip_name, frame_name = results['key'].split(
            '/')  # key example: 000/00000000
        center_frame_idx = int(frame_name)
        num_half_frames = results['num_input_frames'] // 2

        interval = np.random.choice(self.interval_list)
        # ensure not exceeding the borders
        start_frame_idx = center_frame_idx - num_half_frames * interval
        end_frame_idx = center_frame_idx + num_half_frames * interval
        while (start_frame_idx < 0) or (end_frame_idx >= self.frames_per_clip):
            center_frame_idx = np.random.randint(0, self.frames_per_clip)
            start_frame_idx = center_frame_idx - num_half_frames * interval
            end_frame_idx = center_frame_idx + num_half_frames * interval
        frame_name = f'{center_frame_idx:08d}'
        neighbor_list = list(
            range(center_frame_idx - num_half_frames * interval,
                  center_frame_idx + num_half_frames * interval + 1, interval))

        lq_path_root = results['lq_path']
        gt_path_root = results['gt_path']
        lq_path = [
            osp.join(lq_path_root, clip_name, f'{v:08d}.png')
            for v in neighbor_list
        ]
        gt_path = [osp.join(gt_path_root, clip_name, f'{frame_name}.png')]
        results['lq_path'] = lq_path
        results['gt_path'] = gt_path
        results['interval'] = interval

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(interval_list={self.interval_list}, '
                     f'frames_per_clip={self.frames_per_clip})')
        return repr_str
import random
import cv2
from megengine.data.transform import ColorJitter as mge_color_jitter
from edit.utils import imwrite
from ..registry import PIPELINES


def bgr2gray(img, keepdim=True):
    """Convert a BGR image to grayscale image.
    Args:
        img (ndarray): The input image.
        keepdim (bool): If False (by default), then return the grayscale image
            with 2 dims, otherwise 3 dims.
    Returns:
        ndarray: The converted grayscale image.
    """
    out_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if keepdim:
        out_img = out_img[..., None]
    return out_img


@PIPELINES.register_module()
class Bgr2Gray(object):
    def __init__(self, keys, keepdim=True):
        self.keys = keys
        self.keep_dim = keepdim

    def __call__(self, results):
        for key in self.keys:
            if isinstance(results[key], list):
                results[key] = [
                    bgr2gray(v, self.keep_dim) for v in results[key]
                ]
            else:
                results[key] = bgr2gray(results[key], self.keep_dim)
        return results

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string


@PIPELINES.register_module()
class ColorJitter(object):
    def __init__(self, keys, brightness=0.3, contrast=0.3, saturation=0.3, hue=0):
        self.keys = keys
        self.colorjitter = mge_color_jitter(brightness, contrast, saturation, hue)
        # self.nums = 0

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            if isinstance(results[key], list):
                results[key] = [
                    self.colorjitter.apply(v) for v in results[key]
                ]
            else:
                # imwrite(results[key], "./workdirs/{}_origin.png".format(self.nums))
                results[key] = self.colorjitter.apply(results[key])
                # imwrite(results[key], "./workdirs/{}_hou.png".format(self.nums))
                # self.nums += 1
        return results

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string
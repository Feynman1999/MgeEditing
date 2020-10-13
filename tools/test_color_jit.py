import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import random
import numpy as np
from megengine.data.transform import ColorJitter as mge_color_jitter
from edit.utils import imwrite, tensor2img, bgr2ycbcr, imrescale

class ColorJitter(object):
    def __init__(self, keys, brightness=0.3, contrast=0.3, saturation=0.3, hue=0):
        self.keys = keys
        self.colorjitter = mge_color_jitter(brightness, contrast, saturation, hue)

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
                results[key] = self.colorjitter.apply(results[key])
        return results

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string

cj = ColorJitter(keys=["img"])
img = np.ones((100, 100, 3), dtype=np.uint8) * 100
results = {'img':img}
results = cj(results)
imwrite(results['img'], file_path ="./test.png")
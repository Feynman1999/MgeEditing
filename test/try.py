import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from edit.utils import images2video
import megengine.functional as F
import megengine as mge
import numpy as np

def get_xy_ctr_np(score_size):
    fm_height, fm_width = score_size, score_size

    y_list = np.linspace(0., fm_height - 1., fm_height).reshape(1, 1, fm_height, 1)
    y_list = y_list.repeat(fm_width, axis=3)
    x_list = np.linspace(0., fm_width - 1., fm_width).reshape(1, 1, 1, fm_width)
    x_list = x_list.repeat(fm_height, axis=2)
    xy_list = np.concatenate((y_list, x_list), 1)
    xy_ctr = mge.tensor(xy_list.astype(np.float32))
    return xy_ctr

def pixel_un_shuffle(inputs):
    """
        [b,c,h,w]  -> [b,4c,h//2, w//2]
    """
    b,c,h,w = inputs.shape
    x = inputs.transpose(0, 2, 1, 3) # [B, H, C, W]
    x = x.reshape(b, h//2, 2*c, w)
    x = x.transpose(0, 1, 3, 2) # [B, h//2, w, 2*c]
    x = x.reshape(b, h//2, w//2, 4*c)   
    return x.transpose(0, 3, 1, 2)

x = np.arange(0, 8)
x = mge.tensor(x).reshape(1,2,2,2)
print(x)
y = pixel_un_shuffle(x)
print(y)
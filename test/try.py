import os
import sys
from typing import List
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from edit.utils import images2video
import megengine.functional as F
import megengine as mge
import numpy as np
from megengine.module import Conv2d

def get_xy_ctr_np(score_size, score_offset, total_stride):
    """ generate coordinates on image plane for score map pixels (in numpy)
    """
    fm_height, fm_width = score_size, score_size

    y_list = np.linspace(0., fm_height - 1., fm_height).reshape(1, 1, fm_height, 1)
    y_list = y_list.repeat(fm_width, axis=3)
    x_list = np.linspace(0., fm_width - 1., fm_width).reshape(1, 1, 1, fm_width)
    x_list = x_list.repeat(fm_height, axis=2)
    xy_list = score_offset + np.concatenate((y_list, x_list), 1) * total_stride
    xy_ctr = mge.tensor(xy_list.astype(np.float32))
    return xy_ctr

score_size = 125 - 80 + 1 
x_size = 500
total_stride = 4
score_offset = (x_size - 1 - (score_size - 1) * total_stride) / 2

# res = get_xy_ctr_np(score_size, score_offset, 4)

# print(res.shape)

# print(res[0, 0, -5:, -5:])

a = Conv2d(3,1,1,1)
w = a.weight
print(w.shape)
delattr(a, 'weight')
setattr(a, "weight_orig", w)
setattr(a, 'weight', mge.tensor(np.zeros((1,3,1,1))))
# b = a.weight.reshape(3, -1)
# print(b.__class__)

for key, param in a.named_parameters():
    print(key, param.shape)

print("xxxx")

for key, param in a.named_buffers():
    print(key, param.shape)

print("xxxx")

input = mge.tensor(np.random.normal(0,1,(2,3,5,5)))
out = a(input)
print(out.shape)
print(out)
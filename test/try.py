import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from edit.utils import images2video
import megengine.functional as F
import megengine as mge
import numpy as np

H = 24
x = mge.tensor(np.zeros((1, 2, H, H), dtype=np.float32))
kernel =  mge.tensor(np.zeros((2, H, H, 1, 3, 3, 1), dtype=np.float32))
res = F.local_conv2d(x, kernel, (1, 1), (1, 1), (1, 1))

print(res.shape)
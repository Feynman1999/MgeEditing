import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from edit.utils import images2video
import megengine.functional as F
import megengine as mge
import numpy as np


def test(x):
    x[0] = 0


a = np.array([1,1])
print(a)
test(a)
print(a)
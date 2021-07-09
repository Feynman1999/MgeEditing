from megengine import tensor
import megengine.functional as F
import megengine as mge
import numpy as np

x = mge.tensor([0.9,0.009,0.00009])
res = F.softmax(x, axis=0)
print(x)
print(res)

mask = mge.tensor([0.0, 0.0, 1.0])

x = x * (1-mask) + x * mask * (-1e9)
res = F.softmax(x, axis=0)
print(x)
print(res)
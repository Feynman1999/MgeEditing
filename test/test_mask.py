from megengine import tensor
import megengine.functional as F
import megengine as mge
import numpy as np
from megengine.autodiff import GradManager

if __name__ == "__main__":
    gm = GradManager()    
    weight = mge.Parameter(np.random.random((10, 10)))    
    u = mge.tensor(np.random.random((10, 10)))
    gm.attach([weight])
    with gm:
        mask = u > 0.5
        print(mask)
        res = (weight[mask]*2).sum()
        gm.backward(res)
        print(weight.grad)

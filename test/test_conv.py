# version: mge1.1
import megengine.functional as F
import megengine as mge
import numpy as np
from megengine.autodiff import GradManager


if __name__ == "__main__":
    gm = GradManager()    
    groups = 64
    size1, size2 = 250, 160 # 250,16 is ok
    feature = mge.tensor(np.random.random((1, groups, size1, size1)))
    kernel = mge.tensor(np.random.random((groups, 1, 1, size2, size2)))   
    gm.attach([feature, kernel])

    with gm:
        out = F.conv2d(feature, kernel, groups=groups)  # (1, 64, 91, 91)
        loss = out - 10
        gm.backward(loss)

    
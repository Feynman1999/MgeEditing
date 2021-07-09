from megengine import tensor
import megengine.functional as F
import megengine as mge
import numpy as np
from megengine.autodiff import GradManager

if __name__ == "__main__":
    gm = GradManager()    
    weight = mge.Parameter(np.random.random((10, 10)))    
    u = mge.tensor(np.random.random(10, ))
    v = mge.tensor(np.random.random(10, ))
    gm.attach([weight])
    with gm:
        # out = (u *  F.matmul(weight, v)).sum()   # ok ! 
        out = F.dot(u, F.matmul(weight, v))        # error !
        loss = out
        gm.backward(loss)
        print(loss)
        print(weight.grad)

    # if do not gm, both ok
    out1 = (u *  F.matmul(weight, v)).sum()
    out2 = F.dot(u, F.matmul(weight, v))
    print(out1, out2)

    # 结论：
    # with gm时, F.dot报错?
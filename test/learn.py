import megengine as mge
import numpy as np
import megengine.functional as F
from megengine.autodiff import GradManager
from copy import deepcopy
from megengine.optimizer import Optimizer

if __name__ == "__main__":

    # x = mge.tensor([[1., 3., 5.]]) # label: 10
    # label = 10
    # w = mge.tensor([2., 4., 6.]).reshape(3, 1)
    # b = mge.tensor(-1.)
    x = mge.tensor([[22.]]) # label: 10
    label = 10
    w = mge.tensor([6.])
    b = mge.tensor(-1.)
    # y = w*x  +b
    gm = GradManager().attach([w, b])
    lr = 0.001

    for i in range(1000):
        with gm:
            p = F.matmul(x, w) # 【1，1】
            y = p + b
            loss = abs(y - label)
            print("-----------loss: {}".format(loss))
            gm.backward(loss)
        # print(w.grad, b.grad,w,b)
        w -= lr * (w.grad)
        b -= lr * (b.grad-bg)   
        
        
        
import megengine.functional as F
from megengine import Tensor
import megengine
import megengine.module as M
import numpy as np
import math
from ..builder import LOSSES

def safelog(x, eps=None):
    if eps is None:
        eps = np.finfo(x.dtype).eps
    return F.log(F.maximum(x, eps))

@LOSSES.register_module()
class Center_loss(M.Module):
    def __init__(self, alpha = 2, beta = 4):
        super(Center_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, label):
        """
            pred: [B,C,H,W]  C: class nums  H, W: score map size
            label: same to pred, have dealed by gauss function 
        """
        # first cal N (objects) nums
        # 对于每个batch，统计N
        b,c,h,w = label.shape
        peak = (label > 0.99999).astype("float32") # [b,c,h,w]
        peak = peak.reshape(b, -1) # [b, c*h*w]
        pred = pred.reshape(b, -1)
        label = label.reshape(b, -1)
        
        # weight for each sample
        N = F.sum(peak, axis=1, keepdims=True) # [b, 1] 每个batch中物体的数量 
        
        assert float(F.min(N).item()) > 0.000001,  "should contain at least one object in a sample"

        # peak loss
        peak_loss = ((peak * ((1-pred)**self.alpha) * safelog(pred))) # [b, -1]

        # not peak loss
        not_peak_loss = ((1-peak) *  ((1-label)**self.beta) * (pred**self.alpha) * safelog(1-pred)) # [b,-1]

        return -((peak_loss + not_peak_loss) / N).sum() / b


if __name__ == "__main__":
    print("test center loss")
    pred = megengine.tensor([
        [[[0.5, 0.5], [0.5, 0.5]]],
        [[[0.4, 0.4], [0.4, 0.4]]],
    ])
    label = megengine.tensor([
        [[[1, 0.8], [0.8, 0.5]]],
        [[[0.2, 0.4], [0.4, 1]]],
    ])
    loss = Center_loss()
    # 手算：
    # batch1:
    def cal(pred, label):
        return (1-label)**4 * (pred**2) * math.log(1 - pred)
    batch1 = ((0.5)**2 * math.log(0.5) + cal(0.5, 0.8) + cal(0.5, 0.8) + cal(0.5, 0.5)) / 1
    # batch2:
    batch2 = ((0.6)**2 * math.log(0.4) + cal(0.4, 0.2) + cal(0.4, 0.4) + cal(0.4, 0.4)) / 1
    print(loss(pred, label))
    print(batch1)
    print(batch2)
    print(-(batch1 + batch2)/2)
import megengine.functional as F
import megengine.module as M
from megengine.core import Tensor
from ..builder import LOSSES


@LOSSES.register_module()
class BCELoss(M.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        pass

    def forward(self, pred, target, weight=None):
        losses = -1.0 * (target * F.log(pred) + (1.0 - target) * F.log(1 - pred))
        if weight is not None:
            return (losses * weight).sum()
        else:
            return losses.sum()

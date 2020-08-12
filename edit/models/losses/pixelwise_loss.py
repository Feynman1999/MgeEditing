import megengine.module as M
from megengine.functional import loss as mgeloss
from ..builder import LOSSES


@LOSSES.register_module()
class L1Loss(M.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, pred, label):
        return mgeloss.l1_loss(pred = pred, label = label)


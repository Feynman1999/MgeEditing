import megengine.module as M
import megengine.functional as F
from ..builder import LOSSES


@LOSSES.register_module()
class L1Loss(M.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, pred, label):
        return F.l1_loss(pred = pred, label = label)


@LOSSES.register_module()
class CharbonnierLoss(M.Module):
    def __init__(self):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = X - Y
        error = F.sqrt(diff * diff + self.eps)
        loss = F.mean(error)
        return loss


@LOSSES.register_module()
class RSDNLoss(M.Module):
    def __init__(self, a = 1.0, b=1.0, c=1.0):
        super(RSDNLoss, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.charbonnierloss = CharbonnierLoss()

    def forward(self, HR_G, HR_D, HR_S, label, label_D, label_S):
            return self.a * self.charbonnierloss(HR_S, label_S) + \
                self.b * self.charbonnierloss(HR_D, label_D) + \
                self.c * self.charbonnierloss(HR_G, label)


@LOSSES.register_module()
class RSDNLossv2(M.Module):
    def __init__(self):
        super(RSDNLossv2, self).__init__()
        self.charbonnierloss = CharbonnierLoss()

    def forward(self, HR_G, label):
        return self.charbonnierloss(HR_G, label)

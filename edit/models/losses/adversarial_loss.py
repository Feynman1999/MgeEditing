import megengine.functional as F
import megengine.module as M
import megengine
from ..builder import LOSSES

@LOSSES.register_module()
class AdversarialLoss(M.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """
    def __init__(self, losstype='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()
        self.type = losstype
        self.real_label = megengine.tensor(target_real_label)
        self.fake_label = megengine.tensor(target_fake_label)

        if losstype == 'nsgan':
            raise NotImplementedError("nsgan")
        elif losstype == 'lsgan':
            raise NotImplementedError("lsgan")
        elif losstype == 'hinge':
            self.criterion = M.ReLU()
        else:
            raise NotImplementedError("")

    def forward(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()
        else:
            raise NotImplementedError("")
            # labels = F.broadcast_to((self.real_label if is_real else self.fake_label), outputs.shape)
            # loss = self.criterion(outputs, labels)
            # return loss



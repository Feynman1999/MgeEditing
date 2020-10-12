import megengine.functional as F
import megengine.module as M
from megengine.core import Tensor
from ..builder import LOSSES


@LOSSES.register_module()
class IOULoss(M.Module):
    def __init__(self, loc_loss_type='giou'):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None):
        """
            pred: (B*H*W, 4)
            weight: (B*H*W, )
        """
        pred_left = pred[:, 1]
        pred_top = pred[:, 0]
        pred_right = pred[:, 3]
        pred_bottom = pred[:, 2]

        target_left = target[:, 1]
        target_top = target[:, 0]
        target_right = target[:, 3]
        target_bottom = target[:, 2]

        target_aera = (target_left + target_right) * (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * (pred_top + pred_bottom)
                                
        w_intersect = F.minimum(pred_left, target_left) + F.minimum(pred_right, target_right)
        h_intersect = F.minimum(pred_bottom, target_bottom) + F.minimum(pred_top, target_top)
        g_w_intersect = F.maximum(pred_left, target_left) + F.maximum(pred_right, target_right)
        g_h_intersect = F.maximum(pred_bottom, target_bottom) + F.maximum(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.loc_loss_type == 'iou':
            losses = -F.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None:
            return (losses * weight).sum()
        else:
            return losses.sum()
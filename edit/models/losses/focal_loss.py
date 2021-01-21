import megengine.functional as F
from megengine import Tensor
import megengine
import megengine.module as M
import numpy as np
from ..builder import LOSSES

def safelog(x, eps=None):
    if eps is None:
        eps = np.finfo(x.dtype).eps
    return F.log(F.maximum(x, eps))

def softplus(x: Tensor) -> Tensor:
    return F.log(1 + F.exp(-F.abs(x))) + F.relu(x)

def logsigmoid(x: Tensor) -> Tensor:
    return -softplus(-x)

@LOSSES.register_module()
class Focal_loss(M.Module):
    def __init__(self, ignore_label = -1, background = 0, alpha = 0.25, gamma = 2, norm_type="none"):
        super(Focal_loss, self).__init__()
        self.ignore_label = ignore_label
        self.background = background
        self.alpha = alpha
        self.gamma = gamma
        self.norm_type = norm_type

    def forward(self, pred, label):
        return get_focal_loss(pred, label, self.ignore_label, self.background, self.alpha, self.gamma, self.norm_type)


def get_focal_loss(
    logits: Tensor,
    labels: Tensor,
    ignore_label: int = -1,
    background: int = 0,
    alpha: float = 0.25,
    gamma: float = 2,
    norm_type: str = "none",
) -> Tensor:
    r"""Focal Loss for Dense Object Detection:
    <https://arxiv.org/pdf/1708.02002.pdf>
    .. math::
        FL(p_t) = -\alpha_t(1-p_t)^\gamma \log(p_t)  p_t = p if y = 1 else 1-p
    Args:
        logits (Tensor):
            the predicted logits with the shape of :math:`(B, A, C)`
        labels (Tensor):
            the assigned labels of boxes with shape of :math:`(B, A)`
        ignore_label (int):
            the value of ignore class. Default: -1
        background (int):
            the value of background class. Default: 0
        alpha (float):
            parameter to mitigate class imbalance. Default: 0.5
        gamma (float):
            parameter to mitigate easy/hard loss imbalance. Default: 0
        norm_type (str): current support "fg", "none":
            "fg": loss will be normalized by number of fore-ground samples
            "none": not norm
    Returns:
        the calculated focal loss.
    """
    class_range = F.arange(1, logits.shape[2] + 1)  # [1, ]  0 is background
    labels = F.expand_dims(labels, axis=2)  # [B, A, 1]
    # hard_labels = labels > 1e-4
    scores = F.sigmoid(logits)
    
    pos_part = -(1 - scores) ** gamma * logsigmoid(logits)  # logits越大 loss越小
    neg_part = -scores ** gamma * logsigmoid(-logits)  # logits越小 loss 越小
    pos_loss = (labels == class_range).astype("float32") * pos_part * alpha
    neg_loss = (
        (labels != class_range).astype("float32") * (labels != ignore_label).astype("float32") * neg_part * (1 - alpha)
    )

    # loss = (-1) * ((abs(labels - scores))**gamma) * ((1 - labels)*safelog(1 - scores) + labels * safelog(scores))
    # pos_loss = (hard_labels == class_range).astype("float32") * loss * alpha
    # neg_loss = (hard_labels != class_range).astype("float32") * loss * (1 - alpha)
    
    loss = (pos_loss + neg_loss).sum()
    
    if norm_type == "fg":
        # fg_mask = (labels != background) * (labels != ignore_label)
        # return loss / F.maximum(fg_mask.sum(), 1)
        raise NotImplementedError
    elif norm_type == "none":
        return loss
    else:
        raise NotImplementedError
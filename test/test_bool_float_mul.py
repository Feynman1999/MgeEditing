import megengine as mge
import numpy as np
import megengine.functional as F
from megengine.autodiff import GradManager

def xcorr_depthwise(x, kernel):
    """
        x: [B,C,H,W]
        kernel: [B,C,h,w]
    """
    print(x, kernel)    
    batch = int(kernel.shape[0])
    channel = int(kernel.shape[1])
    bc = batch*channel
    if batch != 1000:
        x = x.reshape((1, bc, int(x.shape[2]), int(x.shape[3])))
    kernel = kernel.reshape(bc, 1, 1, int(kernel.shape[2]), int(kernel.shape[3]))
    out = F.conv2d(x, kernel, groups=bc)
    out = out.reshape(batch, channel, int(out.shape[2]), int(out.shape[3]))
    return out

x = mge.tensor(np.random.randn(1,56,200,200))
y = mge.tensor(np.random.randn(1,56,128,128))


# gm = GradManager().attach([w, b])   # 新建一个求导器，绑定需要求导的变量
# with gm:                            # 开始记录计算图

xcorr_depthwise(x, y)


# points = mge.tensor(np.random.randn(1, 2, 46,46))
# points = points.reshape((1,2,46,46))
# gt_bboxes = mge.tensor(np.random.randn(32,4))
# gt_bboxes = F.expand_dims(gt_bboxes, axis=2)
# gt_bboxes = F.expand_dims(gt_bboxes, axis=3)
# # cls_labels
# # 计算四个值以确定是否在内部，由于template比较大，于是缩小bbox为之前的1/4

# gap = (gt_bboxes[:, 2, ...] - gt_bboxes[:, 0, ...]) * 0.25
# up_bound = (points[:, 0, ...] > gt_bboxes[:, 0, ...]) * gap

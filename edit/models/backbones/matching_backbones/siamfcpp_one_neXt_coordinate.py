"""
使用coordinate attention(cvpr 2021)
并使用neXt做为backbone
"""
import numpy as np
import megengine as mge
import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
from edit.models.builder import BACKBONES, build_loss
from edit.models.common.utils import default_init_weights
from edit.models.common import MobileNeXt

def xcorr_depthwise(x, kernel):
    """
        x: [B,C,H,W]
        kernel: [B,C,h,w]
    """
    batch = int(kernel.shape[0])
    channel = int(kernel.shape[1])
    bc = batch*channel
    x = x.reshape((1, bc, int(x.shape[2]), int(x.shape[3])))
    kernel = kernel.reshape(bc, 1, 1, int(kernel.shape[2]), int(kernel.shape[3]))
    out = F.conv2d(x, kernel, groups=bc)
    out = out.reshape(batch, channel, int(out.shape[2]), int(out.shape[3]))
    return out

def get_xy_ctr_np(score_size, score_offset, total_stride):
    """ generate coordinates on image plane for score map pixels (in numpy)
    """
    fm_height, fm_width = score_size, score_size

    y_list = np.linspace(0., fm_height - 1., fm_height).reshape(1, 1, fm_height, 1)
    y_list = y_list.repeat(fm_width, axis=3)
    x_list = np.linspace(0., fm_width - 1., fm_width).reshape(1, 1, 1, fm_width)
    x_list = x_list.repeat(fm_height, axis=2)
    xy_list = score_offset + np.concatenate((y_list, x_list), 1) * total_stride
    xy_ctr = mge.tensor(xy_list.astype(np.float32))
    return xy_ctr

@BACKBONES.register_module()
class SIAMFCPP_one_neXt_coordinate(M.Module):
    def __init__(self, in_cha,
                       channels,
                       loss_cls,
                       loss_bbox,
                       stacked_convs = 3,
                       feat_channels = 48,
                       z_size = 512,
                       x_size = 800,
                       test_z_size = 512,
                       test_x_size = 800,
                       lambda1=0.25,
                       bbox_scale = 0.05,
                       stride = 4,
                       backbone_type = "alexnet"
                       ):
        super(SIAMFCPP_one_neXt_coordinate, self).__init__()
        self.in_cha = in_cha
        self.channels = channels
        self.cls_out_channels = 1
        self.stacked_convs = stacked_convs  # should > 1
        self.feat_channels = feat_channels
        self.total_stride = stride
        self.backbone_type = backbone_type
        self.score_size = x_size // stride - z_size // stride + 1
        self.test_score_size = test_x_size // stride - test_z_size // stride + 1
        self.z_size = z_size
        self.x_size = x_size
        self.test_z_size = test_z_size
        self.test_x_size = test_x_size
        self.lambda1 = lambda1
        self.bbox_scale = bbox_scale

        self.score_offset = (x_size - 1 - (self.score_size - 1) * self.total_stride) / 2  # /2
        self.test_score_offset = (test_x_size - 1 - (self.test_score_size - 1) * self.total_stride) / 2  # /2

        self.fm_ctr = get_xy_ctr_np(self.score_size, self.score_offset, self.total_stride)  # [1, 2, 37, 37]
        self.test_fm_ctr = get_xy_ctr_np(self.test_score_size, self.test_score_offset, self.total_stride)

        self._init_layers() # r_z_k, c_z_k, r_x, c_x, cls_convs, reg_convs, conv_cls, conv_reg, conv_centerness

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

    def _init_layers(self):
        """Initialize layers of the head."""
        if self.total_stride == 4:
            if self.backbone_type == "alexnet":
                self.backbone_sar = AlexNet_stride4(self.in_cha, self.channels)
                self.backbone_opt = AlexNet_stride4(self.in_cha, self.channels)
            else:
                raise NotImplementedError("not implement backbone_type!")
        else:
            raise NotImplementedError("not implement total_stride!")

        self._init_feature_adjust()
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()

    def _init_feature_adjust(self):
        self.r_z_k = M.Sequential(
            M.Conv2d(self.channels, self.feat_channels, 3, 1, 1),
            M.InstanceNorm(self.feat_channels)
        )

        self.c_z_k = M.Sequential(
            M.Conv2d(self.channels, self.feat_channels, 3, 1, 1),
            M.InstanceNorm(self.feat_channels)
        )

        self.r_x = M.Sequential(
            M.Conv2d(self.channels, self.feat_channels, 3, 1, 1),
            M.InstanceNorm(self.feat_channels)
        )

        self.c_x = M.Sequential(
            M.Conv2d(self.channels, self.feat_channels, 3, 1, 1),
            M.InstanceNorm(self.feat_channels)
        )

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = []
        for i in range(self.stacked_convs-1):
            self.cls_convs.append(
                M.ConvRelu2d(self.feat_channels, self.feat_channels, 3, 1, 1)
            )
        self.cls_convs.append(
            M.Sequential(
                M.Conv2d(self.feat_channels, self.feat_channels, 3, 1, 1),
                M.InstanceNorm(self.feat_channels),
                M.ReLU()
            )
        )   
        self.cls_convs = M.Sequential(*self.cls_convs)

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = []
        for i in range(self.stacked_convs-1):
            self.reg_convs.append(
                M.ConvRelu2d(self.feat_channels, self.feat_channels, 3, 1, 1)
            )
        self.reg_convs.append(
            M.Sequential(
                M.Conv2d(self.feat_channels, self.feat_channels, 3, 1, 1),
                M.InstanceNorm(self.feat_channels),
                M.ReLU()
            )
        )
        self.reg_convs = M.Sequential(*self.reg_convs)

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls = M.Conv2d(self.feat_channels, self.cls_out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_reg = M.Conv2d(self.feat_channels, 2, kernel_size=1, stride=1, padding=0)
        
    def head(self, c_out, r_out):
        c_out = self.cls_convs(c_out)
        r_out = self.reg_convs(r_out)
        cls_score = self.conv_cls(c_out)  # [B,1,37,37]
        offsets = self.conv_reg(r_out)
        offsets = F.relu(offsets * self.total_stride + (self.z_size-1)/2)  # [B,2,37,37]
        return [cls_score, offsets]

    def forward(self, sar, optical):
        feat1 = self.backbone_sar(sar)
        feat2 = self.backbone_opt(optical)
        c_z_k = self.c_z_k(feat1)  # 100, 64           
        r_z_k = self.r_z_k(feat1)  # 100, 64      
        c_x = self.c_x(feat2)  # 100, 64      2*48*9*200*200*48
        r_x = self.r_x(feat2)  # 100, 64      2*48*9*200*200*48
        # do depth-wise cross-correlation
        r_out = xcorr_depthwise(r_x, r_z_k)  # [37, 37]
        c_out = xcorr_depthwise(c_x, c_z_k)  # [37, 37]
        # get result
        return self.head(c_out, r_out)

    def get_cls_reg_ctr_targets(self, points, gt_bboxes, bbox_scale = 0.1):
        """
            Compute regression, classification targets for points in multiple images.
            Args:
                points (Tensor): (1, 2, 37, 37).
                gt_bboxes (Tensor): Ground truth bboxes of each image, (B,4), in [tl_x, tl_y, br_x, br_y] format.
            Returns:
                cls_labels (Tensor): Labels. (B, 1, 37, 37)   0 or 1, 0 means background, 1 means in the box.
                bbox_targets (Tensor): BBox targets. (B, 4, 37, 37)  only consider the foreground, for the background should set loss as 0!
        """
        gt_bboxes = F.expand_dims(gt_bboxes, axis=[2,3]) # (B,4,1,1)
        
        # cls_labels
        gap = (gt_bboxes[:, 2, ...] - gt_bboxes[:, 0, ...] + 1) * (1-bbox_scale) / 2  - 0.01  #  应排除的像素数量
        up_bound = (points[:, 0, ...] > gt_bboxes[:, 0, ...] + gap).astype("float32")
        left_bound = (points[:, 1, ...] > gt_bboxes[:, 1, ...] + gap).astype("float32")
        down_bound = (points[:, 0, ...] < gt_bboxes[:, 2, ...] - gap).astype("float32")
        right_bound = (points[:, 1, ...] < gt_bboxes[:, 3, ...] - gap).astype("float32")
        cls_labels = up_bound * left_bound * down_bound * right_bound
        cls_labels = F.expand_dims(cls_labels, axis=1)  # (B, 1, 37, 37)

        # bbox_targets
        up_left = points - gt_bboxes[:, 0:2, ...]  # (B, 2, 37, 37)
        bottom_right = gt_bboxes[:, 2:4, ...] - points
        bbox_targets = F.concat([up_left, bottom_right], axis = 1)  # (B, 4, 37, 37)

        return cls_labels, bbox_targets

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes
            ):
        """Compute loss of the head.

            Args:
                cls_scores (Tensor): [B,1,37,37]
                bbox_preds (Tensor): [B,2,37,37]
                gt_bboxes (Tensor): [B,4], in [tl_x, tl_y, br_x, br_y] format.
                
            Returns:
                dict[str, Tensor]: A dictionary of loss components.
        """
        cls_labels, bbox_targets = self.get_cls_reg_ctr_targets(self.fm_ctr, gt_bboxes, self.bbox_scale)  # (B, 1, 37, 37), (B, 4, 37, 37), (B,1,37,37)
        B,_,H,W = cls_labels.shape
        cls_scores = cls_scores.reshape(B, 1, -1)  # (B, 1, 37*37)
        cls_scores = F.transpose(cls_scores, (0, 2, 1))  # (B, 37*37, 1)
        loss_cls = self.loss_cls(cls_scores, cls_labels.reshape(B, -1)) / (B*H*W)

        bbox_preds = F.concat([bbox_preds, self.z_size -1 - bbox_preds], axis = 1)  # [B,4,37,37]
        bbox_preds = F.transpose(bbox_preds, (0, 2, 3, 1))
        bbox_preds = bbox_preds.reshape(-1, 4)  # (B*37*37, 4)
        bbox_targets = F.transpose(bbox_targets, (0, 2, 3, 1))
        bbox_targets = bbox_targets.reshape(-1, 4)  # (B*37*37, 4)
        loss_reg = self.loss_bbox(bbox_preds, bbox_targets, weight = cls_labels.reshape((B*H*W, ))) / cls_labels.sum()

        loss = (loss_cls + self.lambda1 * loss_reg)
        return loss, loss_cls, loss_reg, cls_labels

    def init_weights(self, pretrained=None, strict=True):
        print("init weights for conv relu")
        default_init_weights(self.cls_convs)
        default_init_weights(self.reg_convs)        


class AlexNet_stride4(M.Module):
    def __init__(self, in_cha, ch=48):
        super(AlexNet_stride4, self).__init__()
        assert ch % 2 ==0, "channel nums should % 2 = 0"
        self.conv1 = M.ConvRelu2d(in_cha, ch//2, kernel_size=3, stride=1, padding=1)  # 2*1*121*400*400*24   929280000        
        self.conv2 = MobileNeXt(ch//2, ch//2, stride = 2, reduction = 4)
        self.conv3 = MobileNeXt(ch//2, ch//2, stride = 1, reduction = 4)
        self.conv4 = MobileNeXt(ch//2, ch, stride = 2, reduction = 4)
        self.conv5 = MobileNeXt(ch, ch, stride = 1, reduction = 4)
        self.conv6 = MobileNeXt(ch, ch, stride = 1, reduction = 4)

    def forward(self, x):
        x = self.conv1(x) # 400, 256
        x = self.conv2(x) # 400, 256
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

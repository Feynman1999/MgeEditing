import megengine as mge
import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
import numpy as np
from edit.models.builder import BACKBONES, build_component, COMPONENTS, build_loss
import time
from megengine.module.normalization import InstanceNorm
from edit.models.common.utils import default_init_weights

def xcorr_depthwise(x, kernel):
    """
        x: [B,C,H,W]
        kernel: [B,C,h,w]
    """
    b, c, h, w = kernel.shape
    gapH = x.shape[2] - h + 1
    gapW = x.shape[3] - w + 1
    res = []
    for i in range(gapH):
        for j in range(gapW):
            # 取x中对应的点乘
            result = x[:, :, i:i + h, j:j + w] * kernel  # [B,C,h,w]
            result = result.reshape(b, c, -1)
            res.append(F.sum(result, axis=2, keepdims=True))  # [B, C, 1]
    res = F.concat(res, axis=2)  # [B,C,5*5]
    return res.reshape(b, c, gapH, gapW)

def get_xy_ctr_np(score_size):
    fm_height, fm_width = score_size, score_size

    y_list = np.linspace(0., fm_height - 1., fm_height).reshape(1, 1, fm_height, 1)
    y_list = y_list.repeat(fm_width, axis=3)
    x_list = np.linspace(0., fm_width - 1., fm_width).reshape(1, 1, 1, fm_width)
    x_list = x_list.repeat(fm_height, axis=2)
    xy_list = np.concatenate((y_list, x_list), 1)
    xy_ctr = mge.tensor(xy_list.astype(np.float32))
    return xy_ctr


@BACKBONES.register_module()
class SIAMFCPP_two_insnorm_deform(M.Module):
    def __init__(self, 
                 in_cha,
                 channels,
                 stacked_convs=3,
                 feat_channels=24,
                 z_size=256,
                 x_size=260,
                 test_z_size=512,
                 test_x_size=520,
                 backbone_type="alexnet"  # alexnet | unet
                 ):
        super(SIAMFCPP_two_insnorm_deform, self).__init__()
        self.in_cha = in_cha
        self.channels = channels
        self.cls_out_channels = 1
        self.stacked_convs = stacked_convs  # should > 1
        self.feat_channels = feat_channels
        self.backbone_type = backbone_type
        self.z_size = z_size
        self.x_size = x_size
        self.test_z_size = test_z_size
        self.test_x_size = test_x_size
        self.score_size = x_size - z_size + 1
        self.test_score_size = test_x_size - test_z_size + 1

        self.fm_ctr = get_xy_ctr_np(self.score_size)  # [1,2,5,5]
        self.test_fm_ctr = get_xy_ctr_np(self.test_score_size)

        self._init_layers()  # r_z_k, c_z_k, r_x, c_x, cls_convs, reg_convs, conv_cls, conv_reg, conv_centerness

    def _init_layers(self):
        """Initialize layers of the head."""
        if self.backbone_type == "alexnet":
            self.backbone_sar = AlexNet(self.in_cha, self.channels)
            self.backbone_opt = AlexNet(self.in_cha, self.channels)
        else:
            raise NotImplementedError("do not know backbone：{}".format(self.backbone_type))

        self._init_feature_adjust()
        self._init_cls_convs()
        self._init_predictor()

    def _init_feature_adjust(self):
        # feature adjustment
        self.c_x = M.Sequential(
                M.Conv2d(self.channels, self.feat_channels, 3, 1, 1),
                InstanceNorm(self.feat_channels),
            )

        self.c_z_k = M.Sequential(
                M.Conv2d(self.channels, self.feat_channels, 3, 1, 1),
                InstanceNorm(self.feat_channels)
            )

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = []
        for i in range(self.stacked_convs - 1):
            self.cls_convs.append(
                M.ConvRelu2d(self.feat_channels, self.feat_channels, 3, 1, 1)
            )
        self.cls_convs.append(
            M.Sequential(
                M.Conv2d(self.feat_channels, self.feat_channels, 3, 1, 1),
                InstanceNorm(self.feat_channels),
                M.ReLU()
            )
        )
        self.cls_convs = M.Sequential(*self.cls_convs)

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls = M.Conv2d(self.feat_channels, self.cls_out_channels, kernel_size=1, stride=1, padding=0)

    def head(self, c_out):
        c_out = self.cls_convs(c_out)
        cls_score = self.conv_cls(c_out)  # [B,1,5,5]
        return cls_score

    def forward(self, sar, optical):
        feat1 = self.backbone_sar(sar)  # b,56,256,256
        feat2 = self.backbone_opt(optical)  # b,56,260,260
        c_z_k = self.c_z_k(feat1)
        c_x = self.c_x(feat2)
        c_out = xcorr_depthwise(c_x, c_z_k)  # [37, 37]
        return self.head(c_out)

    def create_label(self, gt_bboxes):
        gt_bboxes = F.expand_dims(gt_bboxes, axis=[2,3])
        dist = F.sqrt(
            (self.fm_ctr[:, 0, :, :] - gt_bboxes[:, 0, :, :]) ** 2 + (
                    self.fm_ctr[:, 1, :, :] - gt_bboxes[:, 1, :, :]) ** 2)
        labels = F.where(dist <= 3,  # np.where(condition, x, y) 条件为真就返回x 为假则返回y
                         ((-dist / 6) + 1),  #(1 / (0.3* np.pi)) * F.exp(-dist ** 2 / 2)
                         F.zeros_like(dist))
        labels = F.expand_dims(labels, axis=1) #b,1,5,5
        return labels

    def loss(self, cls_scores, gt_bboxes):
        B, _, H, W = cls_scores.shape
        cls_labels = self.create_label(gt_bboxes).reshape(B, -1)  # (B, 1, 5, 5)
        cls_scores = F.sigmoid(cls_scores).reshape(B, -1)
        loss_cls = -1.0* (cls_labels * F.log(cls_scores) + (1.0 - cls_labels) * F.log(1 - cls_scores))
        # 根据labels，凡是大于0.99的加权10倍
        quan = (cls_labels > 0.99).astype("float32") * 1 + 1
        loss_cls = (loss_cls * quan).mean()
        return loss_cls, cls_labels

    def init_weights(self, pretrained=None, strict=True):
        print("init weights for conv relu")
        default_init_weights(self.cls_convs)

class AlexNet(M.Module):
    def __init__(self, in_cha, ch=48):
        super(AlexNet, self).__init__()
        self.groups = 2 # deform groups
        assert ch % (self.groups * 2) == 0, "channel nums should % {} = 0".format(self.groups * 2)
        self.conv1 = M.Sequential(
                M.Conv2d(in_cha, ch//2, 3, 1, 1),
                InstanceNorm(ch//2)
            )
        
        self.offset2 = M.Sequential(
            M.ConvRelu2d(ch//2, ch//2, 3, 1, 1),
            M.Conv2d(ch//2, 3 * self.groups * 9, 3, 1, 1) # mask + offset
        )

        self.conv2 = M.DeformableConv2d(ch//2, ch//2, 3, 1, 1, groups=self.groups)
        
        self.conv3 = M.Sequential(
                M.Conv2d(ch//2, ch, 3, 1, 1),
                InstanceNorm(ch)
            )

        self.offset4 = M.Sequential(
            M.ConvRelu2d(ch, ch, 3, 1, 1),
            M.Conv2d(ch, 3 * self.groups * 9 * 2, 3, 1, 1) # mask + offset
        )

        self.conv4 = M.DeformableConv2d(ch, ch, 3, 1, 1, groups=self.groups * 2)
         
        self.lrelu = M.LeakyReLU(negative_slope=0.2)

        self.initweights()

    def checkoffset(self, offset):
        if F.mean(F.abs(offset)) > 30:
            print("offset abs mean > 30 !!!!!")

    def forward(self, x):
        x = self.lrelu(self.conv1(x))  # 8,28,256,256

        out2 = self.offset2(x)
        offset2 = out2[:, 0:2 * self.groups * 9, ...]
        mask2 =  F.sigmoid(out2[:, 2 * self.groups * 9:, ...])
        x = self.lrelu(self.conv2(x, offset2, mask2))  # 8,28,256,256

        x = self.lrelu(self.conv3(x))  # 8,56,256,256

        out4 = self.offset4(x)
        offset4 = out4[:, 0:2 * self.groups * 9 * 2, ...]
        mask4 =  F.sigmoid(out4[:, 2 * self.groups * 9 * 2:, ...])
        x = self.lrelu(self.conv4(x, offset4, mask4))  # 8,28,256,256

        self.checkoffset(offset2)
        self.checkoffset(offset4)
        return x

    def initweights(self):
        default_init_weights(self.conv1, nonlinearity="leaky_relu", lrelu_value=0.2)
        default_init_weights(self.conv2, nonlinearity="leaky_relu", lrelu_value=0.2)
        default_init_weights(self.conv3, nonlinearity="leaky_relu", lrelu_value=0.2)
        default_init_weights(self.conv4, nonlinearity="leaky_relu", lrelu_value=0.2)
        default_init_weights(self.offset2, scale = 0.1)
        default_init_weights(self.offset4, scale = 0.1)

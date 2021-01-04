import megengine as mge
import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
import numpy as np
from edit.models.common import WeightNet, WeightNet_DW, FilterResponseNorm2d
from edit.models.builder import BACKBONES, build_component, COMPONENTS, build_loss
from megengine.module.normalization import GroupNorm


def xcorr_depthwise(x, kernel):
    """
        x: [B,C,H,W]
        kernel: [B,C,h,w]
    """    
    b,c,h,w = kernel.shape
    gapH = x.shape[2] - h + 1
    gapW = x.shape[3] - w + 1
    res = []
    for i in range(gapH):
        for j in range(gapW):
            # 取x中对应的点乘
            result = x[:, :, i:i+h, j:j+w] * kernel # [B,C,h,w]
            result = result.reshape(b, c, -1)
            res.append(F.sum(result, axis=2, keepdims=True)) # [B, C, 1]
    res = F.concat(res, axis= 2)  # [B,C,5*5]
    return res.reshape(b,c,gapH, gapW)


# def xcorr_depthwise(x, kernel):
#     """
#         x: [B,C,H,W]
#         kernel: [B,C,h,w]
#     """    
#     b,c,h,w = kernel.shape
#     gapH = x.shape[2] - h + 1
#     gapW = x.shape[3] - w + 1
#     res = []
#     for i in range(gapH):
#         for j in range(gapW):
#             # 取x中对应的点乘
#             result = x[:, :, i:i+h, j:j+w] * kernel # [B,C,h,w]
#             result = result.reshape(b, c, -1)
#             res.append(F.sum(result, axis=2, keepdims=True)) # [B, C, 1]
#     res = F.concat(res, axis= 2)  # [B,C,5*5]
#     return res.reshape(b,c,gapH, gapW)


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
class SIAMFCPP_P(M.Module):
    def __init__(self, in_cha,
                       channels,
                       stacked_convs = 3,
                       feat_channels = 24,
                       z_size = 256,
                       x_size = 260,
                       test_z_size = 512,
                       test_x_size = 520,
                       backbone_type = "alexnet",  #  alexnet | Shuffle_weightnet
                       lambda1 = 4
                       ):
        super(SIAMFCPP_P, self).__init__()
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
        self.lambda1 = lambda1

        self.fm_ctr = get_xy_ctr_np(self.score_size) # [1,2,5,5]
        self.test_fm_ctr = get_xy_ctr_np(self.test_score_size)

        self._init_layers() # r_z_k, c_z_k, r_x, c_x, cls_convs, reg_convs, conv_cls, conv_reg, conv_centerness

    def _init_layers(self):
        """Initialize layers of the head."""
        if self.backbone_type == "alexnet":
            self.backbone_sar = AlexNet(self.in_cha, self.channels)
            self.backbone_opt = AlexNet(self.in_cha, self.channels)
        elif self.backbone_type == "Shuffle_weightnet":
            self.backbone_sar = Shuffle_weightnet(self.in_cha, self.channels)
            self.backbone_opt = Shuffle_weightnet(self.in_cha, self.channels)
        else:
            raise NotImplementedError("unknow backbone type")

        self._init_feature_adjust()
        self._init_cls_convs()
        self._init_predictor()

    def _init_feature_adjust(self):
        self.c_z_k = M.Sequential(
            M.Conv2d(self.channels, self.feat_channels, kernel_size=1, stride=1, padding=0),
            GroupNorm(num_groups=8, num_channels=self.feat_channels)
        )
        self.c_x = M.Sequential(
            M.Conv2d(self.channels, self.feat_channels, kernel_size=1, stride=1, padding=0),
            GroupNorm(num_groups=8, num_channels=self.feat_channels)
        )

    def _init_cls_convs(self):
        self.cls_convs = []
        self.cls_convs.append(
            M.Sequential(
                M.Conv2d(self.feat_channels, self.feat_channels, kernel_size=3, stride=1, padding=1),
                GroupNorm(num_groups=8, num_channels=self.feat_channels),
                M.PReLU()
            )
        )
        for i in range(self.stacked_convs-1):
            self.cls_convs.append(
                M.Sequential(
                    M.Conv2d(self.feat_channels, self.feat_channels, kernel_size=1, stride=1, padding=0),
                    GroupNorm(num_groups=8, num_channels=self.feat_channels),
                    M.PReLU()
                )
            )
        self.cls_convs = M.Sequential(*self.cls_convs)

    def _init_predictor(self):
        self.conv_cls = M.Conv2d(self.feat_channels, self.cls_out_channels, kernel_size=1, stride=1, padding=0)

    def head(self, c_out):
        c_out = self.cls_convs(c_out)
        cls_score = self.conv_cls(c_out)
        return cls_score

    def forward(self, sar, optical):
        feat1 = self.backbone_sar(sar)
        feat2 = self.backbone_opt(optical)
        c_z_k = self.c_z_k(feat1)
        c_x = self.c_x(feat2)
        c_out = xcorr_depthwise(c_x, c_z_k)  # [37, 37]
        return self.head(c_out)

    def get_cls_targets(self, gt_bboxes):
        gt_bboxes = F.expand_dims(gt_bboxes, axis=[2,3])  # (B,4,1,1)    关注左上角坐标即可  范围0~4
        dist = F.sqrt((self.fm_ctr[:, 0, :, :] - gt_bboxes[:, 0, :, :]) ** 2 + (self.fm_ctr[:, 1, :, :] - gt_bboxes[:, 1, :, :]) ** 2)
        # label = F.where(dist <= 3,  ((-dist / 6) + 1), F.zeros_like(dist))
        # label = F.expand_dims(label, axis=1) # (B, 1, 5 , 5)
        dist = F.expand_dims(dist, axis = 1)  # (B, 1, 5 , 5)
        return dist

    def loss(self, cls_scores, gt_bboxes):
        # B,2,5,5
        cls_label = self.get_cls_targets(gt_bboxes)  # (B, 1, 5, 5)
        quan = (cls_label < 0.0001).astype("float32") * (self.lambda1 - 1.0) + 1
        X = abs(cls_scores - cls_label)
        loss1 = (F.where(X <1.1, 0.5*(X**2), X-0.5) * quan).mean()
        # cls_scores = F.sigmoid(cls_scores)
        # cls_label = self.get_cls_targets(gt_bboxes)  # (B, 1, 5, 5)
        # quan = (cls_label > 0.99).astype("float32") * (self.lambda1 - 1.0) + 1
        # loss1 = - 1.0 * ((cls_label* F.log(cls_scores) + (1.0 - cls_label) * F.log(1 - cls_scores))*quan).mean()
        return loss1, cls_label

    def init_weights(self, pretrained=None, strict=True):
        # 这里也可以进行参数的load，比如不在之前保存的路径中的模型（预训练好的）
        pass
        # """Init weights for models.
        #
        # Args:
        #     pretrained (str, optional): Path for pretrained weights. If given None, pretrained weights will not be loaded. Defaults to None.
        #     strict (boo, optional): Whether strictly load the pretrained model.
        #         Defaults to True.
        # """
        # if isinstance(pretrained, str):
        #     load_checkpoint(self, pretrained, strict=strict, logger=logger)
        # elif pretrained is None:
        #     pass  # use default initialization
        # else:
        #     raise TypeError('"pretrained" must be a str or None. '
        #                     f'But received {type(pretrained)}.')


class AlexNet(M.Module):
    def __init__(self, in_cha, ch=48):
        super(AlexNet, self).__init__()
        assert ch % 4 ==0, "channel nums should % 2 = 0"
        # self.conv1 = M.conv_bn.ConvBnRelu2d(in_cha, ch//2, kernel_size=3, stride=1, padding=1)
        # self.conv2 = M.conv_bn.ConvBnRelu2d(ch//2, ch, 3, 1, 1)
        self.conv1 = M.Sequential(
            M.Conv2d(in_cha, ch//2, kernel_size=3, stride=1, padding=1),
            GroupNorm(num_groups=4, num_channels=ch//2),
            M.PReLU()
        )
        self.conv2 = M.Sequential(
            M.Conv2d(ch//2, ch, kernel_size=3, stride=1, padding=1),
            GroupNorm(num_groups=8, num_channels=ch),
            M.PReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

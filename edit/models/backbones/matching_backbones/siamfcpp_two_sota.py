import megengine as mge
import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
import numpy as np
from edit.models.builder import BACKBONES, build_component, COMPONENTS, build_loss
import time

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

# def xcorr_depthwise_2(x, kernel):
#     b, c, h, w = kernel.shape
#     _, _, H, W = x.shape
#     x = x.reshape(b*c, H, W)
#     kernel = kernel.reshape(b*c, h, w)
#     # 先算出h维度上的所有


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
class SIAMFCPP_two_sota(M.Module):
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
        super(SIAMFCPP_two_sota, self).__init__()
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
        self.c_z_k = M.ConvBn2d(self.channels, self.feat_channels, 3, 1, 1, momentum=0.9, affine=True,
                                track_running_stats=True)
        self.c_x = M.ConvBn2d(self.channels, self.feat_channels, 3, 1, 1, momentum=0.9, affine=True,
                              track_running_stats=True)

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = []
        for i in range(self.stacked_convs - 1):
            self.cls_convs.append(
                M.ConvRelu2d(self.feat_channels, self.feat_channels, 3, 1, 1)
            )
        self.cls_convs.append(
            M.ConvBnRelu2d(self.feat_channels,
                           self.feat_channels,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           bias=True,
                           momentum=0.9,
                           affine=True,
                           track_running_stats=True)
        )
        self.cls_convs = M.Sequential(*self.cls_convs)

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls = M.ConvBn2d(self.feat_channels, self.cls_out_channels, kernel_size=1, stride=1, padding=0)

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
        loss_cls = -1.0* (cls_labels * F.log(cls_scores) + (1.0 - cls_labels) * F.log(1 - cls_scores)).mean()
        return loss_cls, cls_labels

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
        assert ch % 2 == 0, "channel nums should % 2 = 0"
        self.conv1 = M.conv_bn.ConvBnRelu2d(in_cha, ch // 2, kernel_size=3, stride=1, padding=1) #original
        self.conv2 = CARBBlock(ch // 2)  #original
        self.conv3 = M.conv_bn.ConvBnRelu2d(ch // 2, ch, 1, 1, 0) #original
        self.conv4 = CARBBlock(ch) #original

    def forward(self, x):
        x = self.conv1(x)  # 8,28,256,256
        x = self.conv2(x)  # 8,28,256,256
        x = self.conv3(x)  # 8,56,256,256
        x = self.conv4(x)  # 8,56,256,256
        return x


class CARBBlock(M.Module):
    def __init__(self, channel_num):
        super(CARBBlock, self).__init__()
        self.conv1 = M.Sequential(
            M.conv_bn.ConvBnRelu2d(channel_num, channel_num, kernel_size=3, stride=1, padding=1),
            M.Conv2d(channel_num, channel_num, kernel_size=3, padding=1, stride=1),
        )
        # self.global_average_pooling = nn.AdaptiveAvgPool2d((1,1))  # B,C,H,W -> B,C,1,1
        self.linear = M.Sequential(
            M.Linear(channel_num, channel_num // 2),
            M.ReLU(),
            M.Linear(channel_num // 2, channel_num),
            M.Sigmoid()
        )
        self.conv2 = M.conv_bn.ConvBnRelu2d(channel_num * 2, channel_num, kernel_size=1, stride=1, padding=0)
        self.lrelu = M.LeakyReLU()

    def forward(self, x):
        x1 = self.conv1(x)  # [B, C, H, W]
        w = F.mean(x1, axis=-1, keepdims=False)  # [B,C,H]
        w = F.mean(w, axis=-1, keepdims=False)  # [B,C]
        w = self.linear(w)
        w = F.expand_dims(w, axis=[2,3])
        x1 = F.concat((x1, F.mul(x1, w)), axis=1)  # [B, 2C, H, W]
        del w
        x1 = self.conv2(x1)  # [B, C, H, W]
        return self.lrelu(x + x1)


class CARBBlocks(M.Module):
    def __init__(self, channel_num, block_num):
        super(CARBBlocks, self).__init__()
        self.model = M.Sequential(
            self.make_layer(CARBBlock, channel_num, block_num),
        )

    def make_layer(self, block, channel_num, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(channel_num))
        return M.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

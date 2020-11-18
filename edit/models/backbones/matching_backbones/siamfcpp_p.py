import megengine as mge
import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
import numpy as np
from edit.models.common import WeightNet
from edit.models.builder import BACKBONES, build_component, COMPONENTS, build_loss

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


def get_xy_ctr_np(score_size):
    fm_height, fm_width = score_size, score_size

    y_list = np.linspace(0., fm_height - 1., fm_height).reshape(1, 1, fm_height, 1)
    y_list = y_list.repeat(fm_width, axis=3)
    x_list = np.linspace(0., fm_width - 1., fm_width).reshape(1, 1, 1, fm_width)
    x_list = x_list.repeat(fm_height, axis=2)
    xy_list = np.concatenate((y_list, x_list), 1)
    xy_ctr = mge.tensor(xy_list.astype(np.float32), requires_grad=False)
    return xy_ctr


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
        self.conv2 = M.conv_bn.ConvBnRelu2d(channel_num*2, channel_num, kernel_size=1, stride=1, padding=0)
        self.lrelu = M.LeakyReLU()

    def forward(self, x):
        x1 = self.conv1(x)  # [B, C, H, W]
        w = F.mean(x1, axis = -1, keepdims = False) # [B,C,H]
        w = F.mean(w, axis = -1, keepdims = False) # [B,C]
        w = self.linear(w)
        w = F.add_axis(w, axis = -1)
        w = F.add_axis(w, axis = -1)  # [B,C,1,1]
        x1 = F.concat((x1, F.multiply(x1, w)), axis = 1)  # [B, 2C, H, W]
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


@BACKBONES.register_module()
class SIAMFCPP_P(M.Module):
    def __init__(self, in_cha,
                       channels,
                       loss_cls,
                       stacked_convs = 3,
                       feat_channels = 24,
                       z_size = 256,
                       x_size = 260,
                       test_z_size = 512,
                       test_x_size = 520,
                       backbone_type = "alexnet"  #  alexnet | unet
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

        self.fm_ctr = get_xy_ctr_np(self.score_size) # [1,2,5,5]
        self.test_fm_ctr = get_xy_ctr_np(self.test_score_size)

        self._init_layers() # r_z_k, c_z_k, r_x, c_x, cls_convs, reg_convs, conv_cls, conv_reg, conv_centerness

        self.loss_cls = build_loss(loss_cls)

    def _init_layers(self):
        """Initialize layers of the head."""
        if self.backbone_type == "alexnet":
            self.backbone_sar = AlexNet(self.in_cha, self.channels)
            self.backbone_opt = AlexNet(self.in_cha, self.channels)
        else:
            pass

        self._init_feature_adjust()
        self._init_cls_convs()
        self._init_predictor()

    def _init_feature_adjust(self):
        # feature adjustment
        self.c_z_k = M.ConvBn2d(self.channels, self.feat_channels, 3, 1, 1, momentum=0.9, affine=True, track_running_stats=True)
        self.c_x = M.ConvBn2d(self.channels, self.feat_channels, 3, 1, 1, momentum=0.9, affine=True, track_running_stats=True)

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = []
        for i in range(self.stacked_convs-1):
            self.cls_convs.append(
                M.ConvRelu2d(self.feat_channels, self.feat_channels, 3, 1, 1)
            )
        self.cls_convs.append(
            M.ConvBnRelu2d( self.feat_channels, 
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
        cls_score = self.conv_cls(c_out)  # [B,1,37,37]
        return cls_score

    def forward(self, sar, optical):
        feat1 = self.backbone_sar(sar)
        feat2 = self.backbone_opt(optical)
        c_z_k = self.c_z_k(feat1)
        c_x = self.c_x(feat2)
        c_out = xcorr_depthwise(c_x, c_z_k)  # [37, 37]
        return self.head(c_out)

    def get_cls_targets(self, gt_bboxes):
        B, _ = gt_bboxes.shape
        gt_bboxes = F.add_axis(gt_bboxes, axis=-1)
        gt_bboxes = F.add_axis(gt_bboxes, axis=-1)  # (B,4,1,1)    关注左上角即可  范围0~4
        H = F.abs(self.fm_ctr[:, 0, :, :] - gt_bboxes[:, 0, :, :]) < 0.01  # (B,5,5)
        W = F.abs(self.fm_ctr[:, 1, :, :] - gt_bboxes[:, 1, :, :]) < 0.01
        # 创建一个B*1*25*25的label，
        cls_labels = F.add_axis(H*W, axis=1)  # (B, 1, 5, 5)
        cls_labels.requires_grad = False
        return cls_labels

    def loss(self, cls_scores, gt_bboxes):
        B, _, H, W = cls_scores.shape
        cls_labels = self.get_cls_targets(gt_bboxes)  # (B, 1, 5, 5)
        
        # cls 
        cls_scores = cls_scores.reshape(B, 1, -1)  # (B, 1, 5*5)
        cls_scores = F.dimshuffle(cls_scores, (0, 2, 1))  # (B, 5*5, 1)
        loss_cls = self.loss_cls(cls_scores, cls_labels.reshape(B, -1)) / (B*H*W)

        loss = loss_cls 
        return loss, cls_labels

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
        assert ch % 2 ==0, "channel nums should % 2 = 0"
        self.conv1 = M.conv_bn.ConvBnRelu2d(in_cha, ch//2, kernel_size=3, stride=1, padding=1)
        self.conv2 = CARBBlock(ch//2)
        self.conv3 = M.conv_bn.ConvBnRelu2d(ch//2, ch, 1, 1, 0)
        self.conv4 = CARBBlock(ch)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class ResBlock(M.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, conv_mode="weightnet", usebn=True):
        super(ResBlock, self).__init__()
        self.act = M.PReLU(num_parameters=1, init=0.25)

        m = []
        if conv_mode=="normal":
            m.append(M.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size//2)))
            if usebn:
                m.append(M.BatchNorm2d(out_channels))
            m.append(M.PReLU(num_parameters=1, init=0.25))
            m.append(M.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size//2)))
            if usebn:
                m.append(M.BatchNorm2d(out_channels))
        elif conv_mode=="weightnet":
            m.append(WeightNet(in_channels, out_channels, kernel_size, 1))
            if usebn:
                m.append(M.BatchNorm2d(out_channels))
            m.append(M.PReLU(num_parameters=1, init=0.25))
            m.append(WeightNet(out_channels, out_channels, kernel_size, 1))
            if usebn:
                m.append(M.BatchNorm2d(out_channels))
        else:
            raise NotImplementedError("???")
        self.body = M.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x
        return self.act(res)

class Resnet_weightnet(M.Module):
    def __init__(self, in_cha, ch=48):
        super(Resnet_weightnet, self).__init__()
        assert ch % 2 ==0, "channel nums should % 2 = 0"
        self.conv1 = M.conv_bn.ConvBnRelu2d(in_cha, ch//2, kernel_size=3, stride=1, padding=1)
        self.conv2 = ResBlock(ch//2, ch, 3)
        self.conv3 = ResBlock(ch, ch, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
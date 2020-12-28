import megengine as mge
import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
from edit.models.common import WeightNet, WeightNet_DW, FilterResponseNorm2d
import numpy as np
from edit.models.builder import BACKBONES, build_component, COMPONENTS, build_loss

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
class SIAMFCPP(M.Module):
    def __init__(self, in_cha,
                       channels,
                       loss_cls,
                       loss_bbox,
                       stacked_convs = 3,
                       feat_channels = 24,
                       z_size = 512,
                       x_size = 800,
                       test_z_size = 512,
                       test_x_size = 800,
                       lambda1=2,
                       lambda2=0,
                       bbox_scale = 0.1,
                       stride = 8,
                       backbone_type = "alexnet"  #  alexnet | unet
                       ):
        super(SIAMFCPP, self).__init__()
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
        self.lambda2 = lambda2
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
            elif self.backbone_type == "unet":
                self.backbone_sar = UNet_stride4(self.in_cha, self.channels)
                self.backbone_opt = UNet_stride4(self.in_cha, self.channels)
            else:
                raise NotImplementedError("not implement!")
        elif self.total_stride == 8:
            self.backbone_sar = AlexNet_stride8(self.in_cha, self.channels)
            self.backbone_opt = AlexNet_stride8(self.in_cha, self.channels)
        elif self.total_stride == 2:
            if self.backbone_type == "alexnet":
                self.backbone_sar = AlexNet_stride2(self.in_cha, self.channels)
                self.backbone_opt = AlexNet_stride2(self.in_cha, self.channels)
            elif self.backbone_type == "Shuffle_weight_frn_stride2":
                self.backbone_sar = Shuffle_weight_frn_stride2(self.in_cha, self.channels)
                self.backbone_opt = Shuffle_weight_frn_stride2(self.in_cha, self.channels)
            else:
                raise NotImplementedError("not implement!")
        else:
            pass
        # self.bi = mge.Parameter(value=[0.0])
        # self.si = mge.Parameter(value=[1.0])

        self._init_feature_adjust()
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()

    def _init_feature_adjust(self):
        # feature adjustment
        if "frn" in self.backbone_type:
            self.c_z_k = M.Sequential(
                M.Conv2d(self.channels, self.feat_channels, 3,1,1),
                FilterResponseNorm2d(self.feat_channels)
            )
            # self.c_x = M.ConvBn2d(self.channels, self.feat_channels, 3, 1, 1, momentum=0.9, affine=True, track_running_stats=True)
            self.c_x = M.Sequential(
                M.Conv2d(self.channels, self.feat_channels, 3,1,1),
                FilterResponseNorm2d(self.feat_channels)
            )
            self.r_z_k = M.Sequential(
                M.Conv2d(self.channels, self.feat_channels, 3,1,1),
                FilterResponseNorm2d(self.feat_channels)
            )
            self.r_x = M.Sequential(
                M.Conv2d(self.channels, self.feat_channels, 3,1,1),
                FilterResponseNorm2d(self.feat_channels)
            )
        else:
            self.r_z_k = M.ConvBn2d(self.channels, self.feat_channels, 3, 1, 1, momentum=0.9, affine=True, track_running_stats=True)
            self.c_z_k = M.ConvBn2d(self.channels, self.feat_channels, 3, 1, 1, momentum=0.9, affine=True, track_running_stats=True)
            self.r_x = M.ConvBn2d(self.channels, self.feat_channels, 3, 1, 1, momentum=0.9, affine=True, track_running_stats=True)
            self.c_x = M.ConvBn2d(self.channels, self.feat_channels, 3, 1, 1, momentum=0.9, affine=True, track_running_stats=True)

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = []
        for i in range(self.stacked_convs-1):
            self.cls_convs.append(
                M.ConvRelu2d(self.feat_channels, self.feat_channels, 3, 1, 1)
            )
        if "frn" in self.backbone_type:
            self.cls_convs.append(
                M.Sequential(
                    M.Conv2d(self.feat_channels, self.feat_channels, 3, 1 ,1),
                    FilterResponseNorm2d(self.feat_channels),
                    M.LeakyReLU()
                )
            )  
        else:
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

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = []
        for i in range(self.stacked_convs-1):
            self.reg_convs.append(
                M.ConvRelu2d(self.feat_channels, self.feat_channels, 3, 1, 1)
            )
        if "frn" in self.backbone_type:
            self.reg_convs.append(
                M.Sequential(
                    M.Conv2d(self.feat_channels, self.feat_channels, 3, 1 ,1),
                    FilterResponseNorm2d(self.feat_channels),
                    M.LeakyReLU()
                )
            )
        else:
            self.reg_convs.append(
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
        self.reg_convs = M.Sequential(*self.reg_convs)

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        if "frn" in self.backbone_type:
            self.conv_cls = M.Conv2d(self.feat_channels, self.cls_out_channels, 1, 1 ,0)
            self.conv_reg = M.Conv2d(self.feat_channels, 2, kernel_size=3, stride=1, padding=1)
        else:
            self.conv_cls = M.ConvBn2d(self.feat_channels, self.cls_out_channels, kernel_size=1, stride=1, padding=0)
            self.conv_reg = M.ConvBn2d(self.feat_channels, 2, kernel_size=1, stride=1, padding=0)
        
    def head(self, c_out, r_out):
        c_out = self.cls_convs(c_out)
        r_out = self.reg_convs(r_out)
        # classification score
        cls_score = self.conv_cls(c_out)  # [B,1,37,37]
        # regression
        offsets = self.conv_reg(r_out)
        offsets = F.relu(offsets*self.total_stride + (self.z_size-1)/2)  # [B,2,37,37]
        return [cls_score, offsets]

    def forward(self, sar, optical):
        feat1 = self.backbone_sar(sar)
        feat2 = self.backbone_opt(optical)
        c_z_k = self.c_z_k(feat1)  # 100, 64
        r_z_k = self.r_z_k(feat1)  # 100, 64
        c_x = self.c_x(feat2)  # 100, 64
        r_x = self.r_x(feat2)  # 100, 64
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
        B, _ = gt_bboxes.shape
        gt_bboxes = F.expand_dims(gt_bboxes, axis=2)
        gt_bboxes = F.expand_dims(gt_bboxes, axis=3)  # (B,4,1,1)
        # cls_labels
        # 计算四个值以确定是否在内部，由于template比较大，于是缩小bbox为之前的1/4
        gap = (gt_bboxes[:, 2, ...] - gt_bboxes[:, 0, ...]) * (1-bbox_scale) / 2
        up_bound = (points[:, 0, ...] > gt_bboxes[:, 0, ...]).astype("float32") + gap
        left_bound = (points[:, 1, ...] > gt_bboxes[:, 1, ...]).astype("float32") + gap
        down_bound = (points[:, 0, ...] < gt_bboxes[:, 2, ...]).astype("float32") - gap
        right_bound = (points[:, 1, ...] < gt_bboxes[:, 3, ...]).astype("float32") - gap
        cls_labels = up_bound * left_bound * down_bound * right_bound
        cls_labels = F.expand_dims(cls_labels, axis=1)  # (B, 1, 37, 37)

        # bbox_targets
        # 对于points中的每个坐标，计算偏离情况（这里每个坐标都会计算，所以会有负数）
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
        
        B, _, H, W = cls_scores.shape
        cls_labels, bbox_targets = self.get_cls_reg_ctr_targets(self.fm_ctr, gt_bboxes, self.bbox_scale)  # (B, 1, 37, 37), (B, 4, 37, 37), (B,1,37,37)

        # cls 
        cls_scores = cls_scores.reshape(B, 1, -1)  # (B, 1, 37*37)
        cls_scores = F.transpose(cls_scores, (0, 2, 1))  # (B, 37*37, 1)
        loss_cls = self.loss_cls(cls_scores, cls_labels.reshape(B, -1)) / (B*H*W)

        # reg
        bbox_preds = F.concat([bbox_preds, self.z_size -1 - bbox_preds], axis = 1)  # [B,4,37,37]
        bbox_preds = F.transpose(bbox_preds, (0, 2, 3, 1))
        bbox_preds = bbox_preds.reshape(-1, 4)  # (B*37*37, 4)
        
        bbox_targets = F.transpose(bbox_targets, (0, 2, 3, 1))
        bbox_targets = bbox_targets.reshape(-1, 4)  # (B*37*37, 4)
        loss_reg = self.loss_bbox(bbox_preds, bbox_targets, weight = cls_labels.reshape((B*H*W, ))) / cls_labels.sum()

        loss = (loss_cls + self.lambda1 * loss_reg) 
        return loss, loss_cls, loss_reg, cls_labels

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


class AlexNet_stride8(M.Module):
    def __init__(self, in_cha, ch=48):
        super(AlexNet_stride8, self).__init__()
        assert ch % 2 ==0, "channel nums should % 2 = 0"
        
        self.conv1 = M.conv_bn.ConvBnRelu2d(in_cha, ch//2, kernel_size=11, stride=2, padding=5)
        self.pool1 = M.MaxPool2d(3, 2, 1)
        self.conv2 = M.conv_bn.ConvBnRelu2d(ch//2, ch, 5, 1, 2)
        self.pool2 = M.MaxPool2d(3, 2, 1)
        self.conv3 = M.conv_bn.ConvBnRelu2d(ch, ch*2, 3, 1, 1)
        self.conv4 = M.conv_bn.ConvBnRelu2d(ch*2, ch*2, 3, 1, 1)
        self.conv5 = M.conv_bn.ConvBn2d(ch*2, ch, 3, 1, 1)
        
    def forward(self, x):
        # 800, 512
        x = self.conv1(x) # 400, 256
        x = self.pool1(x) # 200, 128
        x = self.conv2(x) # 200, 128
        x = self.pool2(x) # 100, 64
        x = self.conv3(x) # 100, 64
        x = self.conv4(x) # 100, 64
        x = self.conv5(x) # 100, 64
        return x


class AlexNet_stride4(M.Module):
    def __init__(self, in_cha, ch=48):
        super(AlexNet_stride4, self).__init__()
        assert ch % 2 ==0, "channel nums should % 2 = 0"
        
        self.conv1 = M.conv_bn.ConvBnRelu2d(in_cha, ch//2, kernel_size=11, stride=2, padding=5)
        self.conv2 = M.conv_bn.ConvBnRelu2d(ch//2, ch, 5, 1, 2)
        self.pool1 = M.MaxPool2d(3, 2, 1)
        self.conv3 = M.conv_bn.ConvBnRelu2d(ch, ch, 3, 1, 1)
        self.conv4 = M.conv_bn.ConvBnRelu2d(ch, ch, 3, 1, 1)
        self.conv5 = M.conv_bn.ConvBn2d(ch, ch, 3, 1, 1)
        
    def forward(self, x):
        # 800, 512
        x = self.conv1(x) # 400, 256
        x = self.conv2(x) # 400, 256
        x = self.pool1(x) # 200, 128
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x) # 200, 128
        return x


class ShuffleV2Block(M.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp
        outputs = oup - inp
        self.reduce = M.Conv2d(inp, max(16, inp//16), 1, 1, 0, bias=True)
        self.wnet1 = WeightNet(inp, mid_channels, 1, 1)
        # self.bn1 = M.BatchNorm2d(mid_channels)
        self.bn1 = FilterResponseNorm2d(mid_channels)
        self.wnet2 = WeightNet_DW(mid_channels, ksize, stride)
        # self.bn2 =M.BatchNorm2d(mid_channels)
        self.bn2 = FilterResponseNorm2d(mid_channels)
        self.wnet3 = WeightNet(mid_channels, outputs, 1, 1)
        # self.bn3 = M.BatchNorm2d(outputs)
        self.bn3 = FilterResponseNorm2d(outputs)
        if stride == 2:
            self.wnet_proj_1 = WeightNet_DW(inp, ksize, stride)
            self.bn_proj_1 = M.BatchNorm2d(inp)

            self.wnet_proj_2 = WeightNet(inp, inp, 1, 1)
            self.bn_proj_2 = M.BatchNorm2d(inp)

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x

        x_gap = x.mean(axis=2,keepdims=True).mean(axis=3,keepdims=True)
        x_gap = self.reduce(x_gap)

        x = self.wnet1(x, x_gap)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.wnet2(x, x_gap)
        x = self.bn2(x)

        x = self.wnet3(x, x_gap)
        x = self.bn3(x)
        x = F.relu(x)

        if self.stride == 2:
            x_proj = self.wnet_proj_1(x_proj, x_gap)
            x_proj = self.bn_proj_1(x_proj)
            x_proj = self.wnet_proj_2(x_proj, x_gap)
            x_proj = self.bn_proj_2(x_proj)
            x_proj = F.relu(x_proj)

        return F.concat((x_proj, x), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.shape
        # assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.transpose(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class Shuffle_weight_frn_stride2(M.Module):
    def __init__(self, in_cha, ch=48):
        super(Shuffle_weight_frn_stride2, self).__init__()
        assert ch % 2 ==0, "channel nums should % 2 = 0"
        self.conv1 = M.Sequential(
            M.Conv2d(in_cha, ch//2, kernel_size=5, stride=2, padding=2),
            M.LeakyReLU(),
            FilterResponseNorm2d(num_features=ch//2)
        )
        self.conv2 = ShuffleV2Block(ch//4, ch, ch//2, ksize = 3, stride=1)
        self.conv3 = ShuffleV2Block(ch//2, ch, ch//2, ksize = 3, stride=1)
        self.conv4 = ShuffleV2Block(ch//2, ch, ch//2, ksize = 3, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class AlexNet_stride2(M.Module):
    def __init__(self, in_cha, ch=48):
        super(AlexNet_stride2, self).__init__()
        assert ch % 2 ==0, "channel nums should % 2 = 0"
        
        self.conv1 = M.conv_bn.ConvBnRelu2d(in_cha, ch//2, kernel_size=11, stride=2, padding=5)
        self.conv2 = M.conv_bn.ConvBnRelu2d(ch//2, ch, 5, 1, 2)
        self.conv3 = M.conv_bn.ConvBnRelu2d(ch, ch, 3, 1, 1)
        self.conv4 = M.conv_bn.ConvBnRelu2d(ch, ch, 3, 1, 1)

    def forward(self, x):
        # 800, 512
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class UNet_stride4(M.Module):
    def __init__(self, in_cha, ch=48):
        super(UNet_stride4, self).__init__() 
        assert ch % 4 ==0, "channel nums should % 4 = 0"
        
        self.conv1 = M.Sequential(
            M.conv_bn.ConvBnRelu2d(in_cha, ch//4, kernel_size=5, stride=1, padding=2),
            M.conv_bn.ConvBnRelu2d(ch//4, ch//4, kernel_size=3, stride=1, padding=1),
            M.conv_bn.ConvBnRelu2d(ch//4, ch//4, kernel_size=3, stride=1, padding=1)
        )
        
        self.conv2 = M.Sequential(
            M.conv_bn.ConvBnRelu2d(ch//4, ch//2, kernel_size=3, stride=2, padding=1),
            M.conv_bn.ConvBnRelu2d(ch//2, ch//2, kernel_size=3, stride=1, padding=1)
        )

        self.conv3 = M.Sequential(
            M.conv_bn.ConvBnRelu2d(ch//2, ch, kernel_size=3, stride=2, padding=1),
            M.conv_bn.ConvBnRelu2d(ch, ch, kernel_size=3, stride=1, padding=1)
        )

        self.prs1 = PixelReverseShuffle(scale = 4)
        self.prs2 = PixelReverseShuffle(scale = 2)

        self.aggre = M.conv_bn.ConvBnRelu2d(ch * 4 + ch*2 + ch, ch, kernel_size=1, stride=1, padding=0)
        self.conv4 = M.conv_bn.ConvBnRelu2d(ch, ch, kernel_size=3, stride=1, padding=1)

        # self.conv4 = M.Sequential(
        #     M.conv_bn.ConvBnRelu2d(ch*4, ch*8, kernel_size=3, stride=2, padding=1),
        #     M.conv_bn.ConvBnRelu2d(ch*8, ch*8, kernel_size=3, stride=1, padding=1)
        # )

        # self.conv5 = M.Sequential(
        #     M.conv_bn.ConvBnRelu2d(ch*8, ch*8, kernel_size=3, stride=2, padding=1),
        #     M.conv_bn.ConvBnRelu2d(ch*8, ch*8, kernel_size=3, stride=1, padding=1),
        #     M.conv_bn.ConvBnRelu2d(ch*8, ch*8, kernel_size=3, stride=1, padding=1)
        # )

         
    def forward(self, x):
        x1 = self.conv1(x) 
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x3 = F.concat([self.prs1(x1), self.prs2(x2), x3], axis=1)
        x3 = self.aggre(x3)
        x3 = self.conv4(x3)
        return x3


class PixelReverseShuffle(M.Module):
    def __init__(self, scale=2):
        super(PixelReverseShuffle, self).__init__()
        self.scale = scale

    def forward(self, inputs):
        # N C iH iW
        N, C, iH, iW = inputs.shape
        oH = iH // self.scale
        oW = iW // self.scale
        oC = C * (self.scale ** 2)
        output = inputs.reshape(N, C, self.scale, oH, self.scale, oW)
        output = F.transpose(output, (0, 1, 4, 2, 3, 5))
        # N C oH oW
        output = output.reshape(N, oC, oH, oW)
        return output
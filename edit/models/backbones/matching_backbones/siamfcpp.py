import megengine as mge
import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
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
    x = x.reshape(1, bc, x.shape[2], x.shape[3])
    kernel = kernel.reshape(bc, 1, 1, kernel.shape[2], kernel.shape[3])
    out = F.conv2d(x, kernel, groups=bc)
    out = out.reshape(batch, channel, out.shape[2], out.shape[3])
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
    xy_ctr = mge.tensor(xy_list.astype(np.float32), requires_grad=False)
    return xy_ctr


@BACKBONES.register_module()
class SIAMFCPP(M.Module):
    """
        channels (int): Number of channels in the input feature map.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
    """
    def __init__(self, in_cha,
                       channels,
                       loss_cls,
                       loss_bbox,
                       loss_centerness,
                       stacked_convs = 3,
                       feat_channels = 24,
                       z_size = 512,
                       x_size = 800,
                       lambda1=2,
                       lambda2=0,
                       bbox_scale = 0.1,
                       stride = 8
                       ):
        self.in_cha = in_cha
        self.channels = channels
        self.cls_out_channels = 1
        self.stacked_convs = stacked_convs  # should > 1
        self.feat_channels = feat_channels
        self.total_stride = stride
        self.score_size = x_size // stride - z_size // stride + 1
        self.z_size = z_size
        self.x_size = x_size
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.bbox_scale = bbox_scale

        self.score_offset = (x_size - 1 - (self.score_size - 1) * self.total_stride) / 2  # /2

        self.fm_ctr = get_xy_ctr_np(self.score_size, self.score_offset, self.total_stride)  # [1, 2, 37, 37]

        self._init_layers() # r_z_k, c_z_k, r_x, c_x, cls_convs, reg_convs, conv_cls, conv_reg, conv_centerness

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_centerness = build_loss(loss_centerness)

    def _init_layers(self):
        """Initialize layers of the head."""
        if self.total_stride == 4:
            self.backbone_sar = AlexNet_stride4(self.in_cha, self.feat_channels)
            self.backbone_opt = AlexNet_stride4(self.in_cha, self.feat_channels)
        elif self.total_stride == 8:
            self.backbone_sar = AlexNet_stride8(self.in_cha, self.feat_channels)
            self.backbone_opt = AlexNet_stride8(self.in_cha, self.feat_channels)
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
        self.conv_cls = M.ConvBn2d(self.feat_channels, self.cls_out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_reg = M.ConvBn2d(self.feat_channels, 2, kernel_size=1, stride=1, padding=0)
        self.conv_centerness = M.ConvBn2d(self.feat_channels, 1, kernel_size=1, stride=1, padding=0)

    def head(self, c_out, r_out):
        c_out = self.cls_convs(c_out)
        r_out = self.reg_convs(r_out)

        # classification score
        cls_score = self.conv_cls(c_out)  # [B,1,37,37]
        
        # center-ness score
        ctr_score = self.conv_centerness(r_out)
        ctr_score = F.sigmoid(ctr_score)

        # regression
        offsets = self.conv_reg(r_out)
        offsets = F.relu(offsets*self.total_stride + (self.z_size-1)/2)  # [B,2,37,37]
        
        # bbox decoding
        # bbox = get_box(self.fm_ctr, offsets)  # (B, 2, 37, 37)

        return [cls_score, offsets, ctr_score]

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
                centerness_targets (Tensor): (B, 1, 37, 37)  only consider the foreground, for the background should set loss as 0!
        """
        B, _ = gt_bboxes.shape
        gt_bboxes = F.add_axis(gt_bboxes, axis=-1)
        gt_bboxes = F.add_axis(gt_bboxes, axis=-1)  # (B,4,1,1)
        # cls_labels
        # 计算四个值以确定是否在内部，由于template比较大，于是缩小bbox为之前的1/4
        gap = (gt_bboxes[:, 2, ...] - gt_bboxes[:, 0, ...]) * (1-bbox_scale) / 2
        up_bound = points[:, 0, ...] > gt_bboxes[:, 0, ...] + gap
        left_bound = points[:, 1, ...] > gt_bboxes[:, 1, ...] + gap
        down_bound = points[:, 0, ...] < gt_bboxes[:, 2, ...] - gap
        right_bound = points[:, 1, ...] < gt_bboxes[:, 3, ...] - gap
        cls_labels = up_bound * left_bound * down_bound * right_bound
        cls_labels = F.add_axis(cls_labels, axis=1)  # (B, 1, 37, 37)
        cls_labels.requires_grad = False

        # bbox_targets
        # 对于points中的每个坐标，计算偏离情况（这里每个坐标都会计算，所以会有负数）
        up_left = points - gt_bboxes[:, 0:2, ...]  # (B, 2, 37, 37)
        bottom_right = gt_bboxes[:, 2:4, ...] - points
        bbox_targets = F.concat([up_left, bottom_right], axis = 1)  # (B, 4, 37, 37)
        bbox_targets.requires_grad = False

        # centerness_targets
        up_bottom = F.minimum(up_left[:, 0, ...], bottom_right[:, 0, ...]) / F.maximum(up_left[:, 0, ...], bottom_right[:, 0, ...])
        left_right = F.minimum(up_left[:, 1, ...], bottom_right[:, 1, ...]) / F.maximum(up_left[:, 1, ...], bottom_right[:, 1, ...])
        centerness_targets = F.sqrt(F.abs(up_bottom * left_right))
        centerness_targets = F.add_axis(centerness_targets, axis =1)  # (B,1,37,37)
        centerness_targets.requires_grad = False
        return cls_labels, bbox_targets, centerness_targets

    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes
            ):
        """Compute loss of the head.

            Args:
                cls_scores (Tensor): [B,1,37,37]
                bbox_preds (Tensor): [B,2,37,37]
                centernesses (Tensor): [B,1,37,37]
                gt_bboxes (Tensor): [B,4], in [tl_x, tl_y, br_x, br_y] format.
                
            Returns:
                dict[str, Tensor]: A dictionary of loss components.
        """
        
        B, _, H, W = cls_scores.shape
        cls_labels, bbox_targets, centerness_targets = self.get_cls_reg_ctr_targets(self.fm_ctr, gt_bboxes, self.bbox_scale)  # (B, 1, 37, 37), (B, 4, 37, 37), (B,1,37,37)
        
        # cls 
        cls_scores = cls_scores.reshape(B, 1, -1)  # (B, 1, 37*37)
        cls_scores = F.dimshuffle(cls_scores, (0, 2, 1))  # (B, 37*37, 1)
        loss_cls = self.loss_cls(cls_scores, cls_labels.reshape(B, -1)) / (B*H*W)

        # reg
        bbox_preds = F.concat([bbox_preds, self.z_size -1 - bbox_preds], axis = 1)  # [B,4,37,37]
        bbox_preds = F.dimshuffle(bbox_preds, (0, 2, 3, 1))
        bbox_preds = bbox_preds.reshape(-1, 4)  # (B*37*37, 4)
       
        bbox_targets = F.dimshuffle(bbox_targets, (0, 2, 3, 1))
        bbox_targets = bbox_targets.reshape(-1, 4)  # (B*37*37, 4)
        loss_reg = self.loss_bbox(bbox_preds, bbox_targets, weight = cls_labels.reshape((B*H*W, ))) / cls_labels.sum()

        # center
        loss_ctr = self.loss_centerness(centernesses, centerness_targets, weight = cls_labels) / cls_labels.sum()
        
        loss = (loss_cls + self.lambda1 * loss_reg + self.lambda2 * loss_ctr) 
        return loss, loss_cls, loss_reg, loss_ctr

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
        x = self.conv3(x) # 200, 128
        x = self.conv4(x) # 200, 128
        x = self.conv5(x) # 200, 128
        return x

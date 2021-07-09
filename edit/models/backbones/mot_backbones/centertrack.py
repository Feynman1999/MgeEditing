import numpy as np
import megengine
import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
from edit.models.common import default_init_weights, PoseResNet, gen_gaussian_target, DLASeg, DLASeg_GN
from edit.models.losses import Center_loss
from edit.models.builder import BACKBONES

backbone_spec = {'PoseResNet': PoseResNet, 'DLA': DLASeg, 'DLA_GN': DLASeg_GN}

@BACKBONES.register_module()
class CenterTrack(M.Module):
    def __init__(self,
                 inp_h = 480,
                 inp_w = 480,
                 stride = 2,
                 channels = 32,
                 head_channels = 64,
                 backbone_type = "PoseResNet",
                 num_layers = 18,
                 num_classes = 1,
                 backbone_imagenet_pretrain = False,
                 all_pretrain = False,
                 all_pretrain_path = None,
                 min_overlap = 0.3,
                 fp = 0.1,
                 fn = 0.4):
        super(CenterTrack, self).__init__()
        assert backbone_type in backbone_spec.keys()
        if backbone_type == 'PoseResNet':
            self.backbone_out_c = 256
        else:
            self.backbone_out_c = 64

        support_strides = [2]
        assert stride in support_strides
        assert inp_h % stride == 0 and inp_w % stride == 0

        self.inp_h = inp_h
        self.inp_w = inp_w
        self.stride = stride
        self.backbone_type = backbone_type
        self.channels = channels
        self.head_channels = head_channels
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.backbone_imagenet_pretrain = backbone_imagenet_pretrain
        self.all_pretrain = all_pretrain
        self.all_pretrain_path = all_pretrain_path
        self.min_overlap = min_overlap
        self.fp = fp
        self.fn = fn
        self.hm_disturb = 0.05 # 用于对pre_hm进行扰动的比例

        self.fm_ctr = self.get_fm_ctr(inp_h, inp_w, stride)

        self.base_layer = M.Sequential(
                            M.Conv2d(3, channels, kernel_size=7, stride=1, padding=3, bias=False),
                            M.BatchNorm2d(channels),
                            M.ReLU()
                        ) # 480 -> 240
        self.pre_layer = M.Sequential(
                            M.Conv2d(3, channels, kernel_size=7, stride=1, padding=3, bias=False),
                            M.BatchNorm2d(channels),
                            M.ReLU()
                        )
        self.hm_layer = M.Sequential(
                            M.Conv2d(1, channels, kernel_size=7, stride=1, padding=3, bias=False),
                            M.BatchNorm2d(channels),
                            M.ReLU()
                        )

        self.init_backbone()
        self.init_heads()
        self.center_loss = Center_loss(alpha = 2, beta = 4)
        self.init_weights(pretrained = self.all_pretrain)

    def init_backbone(self):
        select_backbone = backbone_spec[self.backbone_type]
        self.backbone = select_backbone(num_layers = self.num_layers, inp = self.channels, pretrained = self.backbone_imagenet_pretrain)

    def init_heads(self):
        """
            init 3 kinds of heads: heat map, HW and motion
        """
        self.head_heatmap = M.Sequential(
            M.Conv2d(self.backbone_out_c, self.head_channels, 3, 1, 1, bias=True),
            M.ReLU(),
            M.Conv2d(self.head_channels, self.num_classes, 3, 1, 1, bias=True)
        )
        self.head_hw = M.Sequential(
            M.Conv2d(self.backbone_out_c, self.head_channels, 3, 1, 1, bias=True),
            M.ReLU(),
            M.Conv2d(self.head_channels, 2, 3, 1, 1, bias=True)
        )
        self.head_motion = M.Sequential(
            M.Conv2d(self.backbone_out_c, self.head_channels, 3, 1, 1, bias=True),
            M.ReLU(),
            M.Conv2d(self.head_channels, 2, 3, 1, 1, bias=True)
        )

    def forward(self, x, pre_img = None, pre_hm = None):
        x = self.base_layer(x)
        if pre_img is not None:
            pre_img = self.pre_layer(pre_img)
            x = x + pre_img
        if pre_hm is not None:
            pre_hm = self.hm_layer(pre_hm)
            x = x + pre_hm

        x = self.backbone(x)

        heatmap = self.head_heatmap(x)
        hw = self.head_hw(x)
        motion = self.head_motion(x)
        heatmap = F.sigmoid(heatmap)
        return heatmap, hw, motion

    def get_fm_ctr(self, inph, inpw, stride):
        fm_height, fm_width = inph // stride, inpw // stride # 240, 240
        y_list = np.linspace(0., fm_height - 1., fm_height).reshape(1, 1, fm_height, 1)
        y_list = y_list.repeat(fm_width, axis=3)
        x_list = np.linspace(0., fm_width - 1., fm_width).reshape(1, 1, 1, fm_width)
        x_list = x_list.repeat(fm_height, axis=2)
        xy_list = (self.stride - 1.0)/2 + np.concatenate((x_list, y_list), 1) * self.stride
        xy_list = xy_list.astype(np.float32)
        return xy_list

    def loss_hw(self, pred_hw, gt_hw, gt_mask):
        """
            pred_hw: b,2,h,w
            gt_hw: b,2,h,w
            gt_mask: b,1,h,w
            注意要batchwise的做，因为每个sample的框数量不一样
        """
        bs, _, _, _ = pred_hw.shape
        loss = gt_mask * F.abs(pred_hw - gt_hw) # [b,2,h,w]
        loss = loss.reshape(bs, -1)
        loss = F.sum(loss, axis=1, keepdims=True) / 2 # [b, 1]
        gt_mask = gt_mask.reshape(bs, -1)
        gt_mask = F.sum(gt_mask, axis=1, keepdims=True) # [b, 1]
        loss = loss / gt_mask
        loss = loss.sum() / bs
        return loss

    def loss_motion(self, pred_motion, gt_motion, gt_mask):
        """
            对于给定的前一帧和当前帧，根据同一id的位置变化，来学习motion
        """
        bs, _, _, _ = pred_motion.shape
        loss = gt_mask * F.abs(pred_motion - gt_motion) # [b,2,h,w]
        loss = loss.reshape(bs, -1)
        loss = F.sum(loss, axis=1, keepdims=True) / 2 # [b, 1]
        gt_mask = gt_mask.reshape(bs, -1)
        gt_mask = F.sum(gt_mask, axis=1, keepdims=True) # [b, 1]
        loss = loss / gt_mask
        loss = loss.sum() / bs
        return loss

    def get_loss(self, pred_heatmap, pred_hw, pred_motion, gt_bboxes, gt_labels, loss_weight, pre_gt_bboxes=None, pre_gt_labels = None):
        """
            given pre hm and now bbox, cal loss
        """
        gt_hms, gt_hw, gt_mask, gt_motion = self.get_targets(gt_bboxes, gt_labels, pre_gt_bboxes, pre_gt_labels)
        loss_hms = self.center_loss(pred_heatmap, gt_hms)
        loss_hw = self.loss_hw(pred_hw, gt_hw, gt_mask)
        loss_motion = self.loss_motion(pred_motion, gt_motion, gt_mask)
        weight_hms = loss_weight['hms']
        weight_hw = loss_weight['hw']
        weight_motion = loss_weight['motion']
        total_loss = weight_hms * loss_hms + weight_hw * loss_hw + weight_motion * loss_motion
        return [loss_hms, loss_hw, loss_motion, total_loss]

    def get_gaussian_radius(self, det_size, min_overlap):
        """
            for centernet, only have one case
            refer to https://github.com/princeton-vl/CornerNet/blob/e5c39a31a8abef5841976c8eab18da86d6ee5f9a/sample/utils.py
        """
        assert min_overlap > 0.05 and min_overlap < 0.95
        height, width = det_size

        a1  = 1
        b1  = (height + width)
        c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1  = (b1 - sq1) / (2 * a1)
        return r1

    def map_to_feat(self, x_or_y, w_or_h, factor = 1):
        """
            加一定概率的扰动,根据 self.hm_disturb
        """
        x_or_y_int = x_or_y + np.random.randn() * self.hm_disturb * w_or_h * factor
        x_or_y_int = (x_or_y_int - (self.stride-1.0)/2) / self.stride
        x_or_y_int = int(x_or_y_int + 0.5)
        return x_or_y_int

    def get_test_pre_hm(self, pre_gt_bboxes):
        bs = len(pre_gt_bboxes)
        assert bs == 1
        _, _, feat_h, feat_w = self.fm_ctr.shape
        pre_hms = F.zeros((bs, self.num_classes, feat_h, feat_w))
        for batch_id in range(bs):
            gt_bbox = pre_gt_bboxes[batch_id] # [S, 4]
            center_x = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2 # w
            center_y = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2 # h
            for gt_id in range(center_x.shape[0]):
                # get radius
                box_w = gt_bbox[gt_id][2] - gt_bbox[gt_id][0]
                box_h = gt_bbox[gt_id][3] - gt_bbox[gt_id][1]
                scale_box_w = box_w / self.stride
                scale_box_h = box_h / self.stride
                # 在给定hw的情况下，决定一个radius，使得iou至少为min_overlap
                radius = self.get_gaussian_radius([scale_box_h, scale_box_w], self.min_overlap)
                radius = max(0, int(radius))
                ind = 0

                ctx = center_x[gt_id]
                cty = center_y[gt_id]
                ctx1_int = self.map_to_feat(ctx, box_w, factor=0)
                cty1_int = self.map_to_feat(cty, box_h, factor=0)
                pre_hms[batch_id, ind] = gen_gaussian_target(pre_hms[batch_id, ind], [ctx1_int, cty1_int], radius)

        pre_hms = F.vision.interpolate(pre_hms, scale_factor=self.stride, align_corners=False)
        return pre_hms

    def get_pre_hm(self, pre_gt_bboxes, pre_gt_labels):
        """
            根据pre_gt_bboxes，生成pre hm，这里使用feat进行bilinear上采样到输入图片的尺寸
            pre_gt_bboxes: list of ndarray [[S, 4] ....]  the 4 is [tl_x, tl_y, br_x, br_y]  float64
            pre_gt_labels: list of ndarray  [[S, 2] ....]  the 2 is [class, id]               int64

            # 对于gt_bboxes1有三种增强方法 按照下面的顺序执行
            1.一定概率消失, 什么也不加       self.fn
            2.按照正太分布随机移动框的位置（移动后不超边界）
            3.一定概率在自己的周围再加一个框  self.fp
        """
        bs = len(pre_gt_bboxes)
        _, _, feat_h, feat_w = self.fm_ctr.shape
        pre_hms = F.zeros((bs, self.num_classes, feat_h, feat_w))
        for batch_id in range(bs):
            gt_bbox = pre_gt_bboxes[batch_id] # [S, 4]
            gt_label = pre_gt_labels[batch_id] # [S, 2]
            center_x = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2 # w
            center_y = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2 # h
            for gt_id in range(center_x.shape[0]):
                if np.random.random() < self.fn:
                    continue
                # get radius
                box_w = gt_bbox[gt_id][2] - gt_bbox[gt_id][0]
                box_h = gt_bbox[gt_id][3] - gt_bbox[gt_id][1]
                scale_box_w = box_w / self.stride
                scale_box_h = box_h / self.stride
                # 在给定hw的情况下，决定一个radius，使得iou至少为min_overlap
                radius = self.get_gaussian_radius([scale_box_h, scale_box_w], self.min_overlap)
                radius = max(0, int(radius))
                ind = gt_label[gt_id][0]

                ctx = center_x[gt_id]
                cty = center_y[gt_id]
                ctx1_int = self.map_to_feat(ctx, box_w)
                cty1_int = self.map_to_feat(cty, box_h)
                pre_hms[batch_id, ind] = gen_gaussian_target(pre_hms[batch_id, ind], [ctx1_int, cty1_int], radius)
                # 一定概率再生成一个
                if np.random.random() < self.fp:
                    ctx2_int = self.map_to_feat(ctx, box_w, factor=2)
                    cty2_int = self.map_to_feat(cty, box_h, factor=2)
                    pre_hms[batch_id, ind] = gen_gaussian_target(pre_hms[batch_id, ind], [ctx2_int, cty2_int], radius)

        pre_hms = F.vision.interpolate(pre_hms, scale_factor=self.stride, align_corners=False)
        return pre_hms

    def get_targets(self, gt_bboxes, gt_labels, pre_gt_bboxes = None, pre_gt_labels = None):
        """
            gt_bboxes: list of ndarray  [[S, 4] ....]  the 4 is [tl_x, tl_y, br_x, br_y]  float64
            gt_labels: list of ndarray  [[S, 2] ....]  the 2 is [class, id]               int64
 
            return: (gt_hms, gt_hw, gt_mask)
            gt_hms: [B, classes, h, w]
            gt_hw: [B, 2, h, w]
            gt_mask: [B, 1, h, w]
        """
        assert len(gt_bboxes) == len(gt_labels)
        bs = len(gt_bboxes)
        _, _, feat_h, feat_w = self.fm_ctr.shape

        gt_hms = F.zeros((bs, self.num_classes, feat_h, feat_w))
        gt_hw = F.zeros((bs, 2, feat_h, feat_w))
        gt_mask = F.zeros((bs, 1, feat_h, feat_w))
        gt_motion = None
        
        if pre_gt_bboxes is not None:
            assert pre_gt_labels is not None
            assert len(pre_gt_labels) == len(pre_gt_bboxes) and len(pre_gt_bboxes) == len(gt_bboxes)
            gt_motion = F.zeros((bs, 2, feat_h, feat_w))

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id] # [S, 4]
            gt_label = gt_labels[batch_id] # [S, 2]
            # 算出每一个bbox的中心位置
            center_x = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2 # w
            center_y = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2 # h

            if pre_gt_bboxes is not None:
                # 构造一个字典根据id可以查到上一帧有没有以及有的话，中心位置是多少(小数坐标)
                id_map = {}
                pre_gt_bbox = pre_gt_bboxes[batch_id]
                pre_gt_label = pre_gt_labels[batch_id]
                pre_center_x = (pre_gt_bbox[:, 0] + pre_gt_bbox[:, 2]) / 2 # w
                pre_center_y = (pre_gt_bbox[:, 1] + pre_gt_bbox[:, 3]) / 2 # h
                for gt_id in range(pre_center_x.shape[0]):
                    ID = pre_gt_label[gt_id][1] # ID
                    ctx = pre_center_x[gt_id]
                    ctx = (ctx - (self.stride-1.0)/2) / self.stride
                    cty = pre_center_y[gt_id]
                    cty = (cty - (self.stride-1.0)/2) / self.stride
                    id_map[ID] = (ctx, cty) # 中心在feat中的位置

            for gt_id in range(center_x.shape[0]):
                # get ctx_int, cty_int, 根据gt_centers，找到最接近的在feat中的坐标（可能会出现距离多个点一样的情况）
                ctx = center_x[gt_id]
                ctx = (ctx - (self.stride-1.0)/2) / self.stride
                ctx_int = int(ctx + 0.5)
                cty = center_y[gt_id]
                cty = (cty - (self.stride-1.0)/2) / self.stride
                cty_int = int(cty + 0.5)
                # get radius
                scale_box_w = (gt_bbox[gt_id][2] - gt_bbox[gt_id][0]) / self.stride
                scale_box_h = (gt_bbox[gt_id][3] - gt_bbox[gt_id][1]) / self.stride
                # 在给定hw的情况下，决定一个radius，使得iou至少为min_overlap
                radius = self.get_gaussian_radius([scale_box_h, scale_box_w], self.min_overlap)
                radius = max(0, int(radius))
                ind = gt_label[gt_id][0]
                gt_hms[batch_id, ind] = gen_gaussian_target(gt_hms[batch_id, ind], [ctx_int, cty_int], radius)

                gt_hw[batch_id, 0, cty_int, ctx_int] = scale_box_w
                gt_hw[batch_id, 1, cty_int, ctx_int] = scale_box_h

                gt_mask[batch_id, 0, cty_int, ctx_int] = 1.0                

                # 如果当前id在上一帧中也有，则看其位置在哪里，构造gt_motion的值，如果没有则置motion为0（或者100？）
                if gt_motion is not None:
                    # 拿当前id去字典中找
                    ID = gt_label[gt_id][1]
                    if ID in id_map.keys():
                        # 前一个位置减后一个位置
                        motion = id_map[ID]
                        gt_motion[batch_id, 0, cty_int, ctx_int] = motion[0] - ctx
                        gt_motion[batch_id, 1, cty_int, ctx_int] = motion[1] - cty
                    else:
                        pass # 默认就是0

        return gt_hms, gt_hw, gt_mask, gt_motion

    def init_weights(self, pretrained):
        if pretrained:
            assert self.all_pretrain_path is not None
            assert ".mge" in self.all_pretrain_path
            print("loading pretrained model for all module 🤡🤡🤡🤡🤡🤡...")
            state_dict = megengine.load(self.all_pretrain_path)
            self.load_state_dict(state_dict, strict=True)
        else:
            default_init_weights(self.base_layer)
            default_init_weights(self.pre_layer)
            default_init_weights(self.hm_layer)
            default_init_weights(self.head_heatmap)
            default_init_weights(self.head_hw)
            default_init_weights(self.head_motion)
            
            def set_bias(m):
                assert isinstance(m, M.Conv2d)
                M.init.fill_(m.bias, -2.19)

            set_bias(self.head_heatmap[-1])
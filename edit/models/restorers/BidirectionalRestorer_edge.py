import os
import time
import numpy as np
import megengine.distributed as dist
import megengine as mge
import megengine.functional as F
from megengine.autodiff import GradManager
from edit.core.hook.evaluation import psnr, ssim
from edit.utils import imwrite, tensor2img, bgr2ycbcr, img_multi_padding, img_de_multi_padding, ensemble_forward, ensemble_back
from edit.utils import img_multi_padding, img_de_multi_padding, flow_to_image
from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS
from tqdm import tqdm

def get_bilinear_up(image):
    B,T,C,h,w = image.shape
    image = image.reshape(-1, C,h,w)
    biup = F.nn.interpolate(image, scale_factor=4)
    return biup.reshape(B,T,C,4*h, 4*w) 

def get_bilinear_s_e(image):
    B,T,C,h,w = image.shape
    image = image.reshape(-1, C,h,w)
    down = F.nn.interpolate(image, scale_factor=0.25)
    # print(down.shape)
    down_up = F.nn.interpolate(down, scale_factor=4)
    return down_up, image - down_up

def train_generator_batch(image, label, *, gm, netG, netloss):
    B,T,_,h,w = image.shape
    biup = get_bilinear_up(image)
    structure, edge = get_bilinear_s_e(image)
    # structure_label, edge_label = get_bilinear_s_e(label)
    netG.train()
    with gm:
        res = []
        forward_hiddens = []
        backward_hiddens = []
        s_feature, e_feature, image_feature = netG.s_e_rgb(structure.reshape(B*T, 3, h, w), edge.reshape(B*T, 3, h, w), image.reshape(B*T, 3, h, w))
        s_feature = s_feature.reshape(B, T, -1, h, w)
        e_feature = e_feature.reshape(B, T, -1, h, w)
        image_feature = image_feature.reshape(B, T, -1, h, w)
        # T=0
        hidden_s = F.zeros((B, netG.hidden_channels, h, w))
        hidden_e = F.zeros((B, netG.hidden_channels, h, w))
        for i in range(T):
            now_frame = image[:, i, ...]
            if i==0:
                flow = netG.flownet(now_frame, now_frame)
            else:
                flow = netG.flownet(now_frame, image[:, i-1, ...])
            hidden_s, hidden_e = netG.aggr(hidden_s, hidden_e, flow, s_feature[:, i, ...], e_feature[:, i, ...])
            forward_hiddens.append(F.concat([hidden_e, hidden_s], axis=1))
        # cal backward hiddens
        hidden_s = F.zeros((B, netG.hidden_channels, h, w))
        hidden_e = F.zeros((B, netG.hidden_channels, h, w))
        for i in range(T-1, -1, -1):
            now_frame = image[:, i, ...]
            if i==(T-1):
                flow = netG.flownet(now_frame, now_frame)
            else:
                flow = netG.flownet(now_frame, image[:, i+1, ...])
            hidden_s, hidden_e = netG.aggr(hidden_s, hidden_e, flow, s_feature[:, i, ...], e_feature[:, i, ...])
            backward_hiddens.append(F.concat([hidden_e, hidden_s], axis=1))
        # do upsample for all frames
        for i in range(T):
            res.append(netG.do_upsample(forward_hiddens[i], backward_hiddens[T-i-1], image_feature[:, i, ...]))
        res = F.stack(res, axis = 1) # [B,T,3,H,W]
        loss = netloss(res+biup, label)
        gm.backward(loss)
        if dist.is_distributed():
            loss = dist.functional.all_reduce_sum(loss) / dist.get_world_size()
    return loss

def test_generator_batch(image, *, netG):
    B,T,_,h,w = image.shape
    biup = get_bilinear_up(image)
    structure, edge = get_bilinear_s_e(image)
    netG.eval()
    res = []
    forward_hiddens = []
    backward_hiddens = []
    s_feature, e_feature, image_feature = netG.s_e_rgb(structure.reshape(B*T, 3, h, w), edge.reshape(B*T, 3, h, w), image.reshape(B*T, 3, h, w))
    s_feature = s_feature.reshape(B, T, -1, h, w)
    e_feature = e_feature.reshape(B, T, -1, h, w)
    image_feature = image_feature.reshape(B, T, -1, h, w)
    # T=0
    hidden_s = F.zeros((B, netG.hidden_channels, h, w))
    hidden_e = F.zeros((B, netG.hidden_channels, h, w))
    for i in tqdm(range(T)):
        now_frame = image[:, i, ...]
        if i==0:
            flow = netG.flownet(now_frame, now_frame)
        else:
            flow = netG.flownet(now_frame, image[:, i-1, ...])
        hidden_s, hidden_e = netG.aggr(hidden_s, hidden_e, flow, s_feature[:, i, ...], e_feature[:, i, ...])
        forward_hiddens.append(F.concat([hidden_e, hidden_s], axis=1))
    # cal backward hiddens
    hidden_s = F.zeros((B, netG.hidden_channels, h, w))
    hidden_e = F.zeros((B, netG.hidden_channels, h, w))
    for i in tqdm(range(T-1, -1, -1)):
        now_frame = image[:, i, ...]
        if i==(T-1):
            flow = netG.flownet(now_frame, now_frame)
        else:
            flow = netG.flownet(now_frame, image[:, i+1, ...])
        hidden_s, hidden_e = netG.aggr(hidden_s, hidden_e, flow, s_feature[:, i, ...], e_feature[:, i, ...])
        backward_hiddens.append(F.concat([hidden_e, hidden_s], axis=1))
    # do upsample for all frames
    for i in tqdm(range(T)):
        res.append(netG.do_upsample(forward_hiddens[i], backward_hiddens[T-i-1], image_feature[:, i, ...]))
    res = F.stack(res, axis = 1) # [B,T,3,H,W]
    return res + biup

epoch_dict = {}

def adjust_learning_rate(optimizer, epoch):
    if epoch>=100 and epoch % 1 == 0  and epoch_dict.get(epoch, None) is None:
        epoch_dict[epoch] = True
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * 0.7
        print("adjust lr! , now lr: {}".format(param_group["lr"]))
    
@MODELS.register_module()
class BidirectionalRestorer_edge(BaseModel):
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self, generator, pixel_loss, train_cfg=None, eval_cfg=None, pretrained=None, Fidelity_loss=None):
        super(BidirectionalRestorer_edge, self).__init__()

        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        # generator
        self.generator = build_backbone(generator)
        # loss
        self.pixel_loss = build_loss(pixel_loss)

        if Fidelity_loss:
            self.Fidelity_loss = build_loss(Fidelity_loss)
        else:
            self.Fidelity_loss = None

        # load pretrained
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        self.generator.init_weights(pretrained)

    def train_step(self, batchdata, now_epoch):
        LR_tensor = mge.tensor(batchdata['lq'], dtype="float32")
        HR_tensor = mge.tensor(batchdata['gt'], dtype="float32")
        loss = train_generator_batch(LR_tensor, HR_tensor, gm=self.gms['generator'], netG=self.generator, netloss=self.pixel_loss)
        adjust_learning_rate(self.optimizers['generator'], now_epoch)
        self.optimizers['generator'].step()
        self.optimizers['generator'].clear_grad()
        return loss

    def get_img_id(self, key):
        assert isinstance(key, str)
        return int(key.split("/")[-1][:-4])

    def test_step(self, batchdata, **kwargs):
        # 在最后一帧时，统一进行处理
        lq = batchdata['lq']  # [1,3,3,h,w]
        gt = batchdata['gt']  # [1,3,4*h,4*w]
        lq_paths = [item[0] for item in batchdata['lq_path']] # 3
        now_id = self.get_img_id(lq_paths[1]) # 1对应中间帧
        if now_id==0:
            print("first frame: {}".format(lq_paths[1]))
            self.LR_list = []
            self.HR_list = []

        # pad lq
        _ ,_ ,origin_H, origin_W = lq[:, 1, ...].shape
        lq = img_multi_padding(lq[:, 1, ...], padding_multi=self.eval_cfg.multi_pad, pad_method = "edge") #  edge  constant
        self.LR_list.append(mge.tensor(lq, dtype="float32"))  # [1,3,h,w]
        self.HR_list.append(gt) # numpy

        if now_id == 99:
            # 计算所有帧
            print("start to forward all frames and eval....")
            if self.eval_cfg.gap == 1:
                self.HR_G = test_generator_batch(F.stack(self.LR_list, axis=1), netG=self.generator)
            elif self.eval_cfg.gap == 2:
                self.HR_G_1 = test_generator_batch(F.stack(self.LR_list[::2], axis=1), netG=self.generator)
                self.HR_G_2 = test_generator_batch(F.stack(self.LR_list[1::2], axis=1), netG=self.generator) # [B,T,C,H,W]
                # 交叉组成HR_G
                res = []
                _,T1,_,_,_ = self.HR_G_1.shape
                _,T2,_,_,_ = self.HR_G_2.shape
                assert T1 == T2
                for i in range(T1):
                    res.append(self.HR_G_1[:, i, ...])
                    res.append(self.HR_G_2[:, i, ...])
                self.HR_G = F.stack(res, axis=1) # [B,T,C,H,W]
            else:
                raise NotImplementedError("do not support eval&test gap value")
            
            scale = self.generator.upscale_factor
            self.HR_G = img_de_multi_padding(self.HR_G.numpy(), origin_H=origin_H * scale, origin_W=origin_W * scale) # depad for HR_G   [B,T,C,H,W]
        
        return now_id == 99

    def cal_for_eval(self, gathered_outputs, gathered_batchdata):
        if gathered_outputs:
            crop_border = self.eval_cfg.crop_border
            assert len(self.HR_list) == 100
            res = []
            for i in range(len(self.HR_list)):
                G = tensor2img(self.HR_G[0, i, ...], min_max=(0, 1))
                gt = tensor2img(self.HR_list[i][0], min_max=(0, 1))
                eval_result = dict()
                for metric in self.eval_cfg.metrics:
                    eval_result[metric+"_RGB"] = self.allowed_metrics[metric](G, gt, crop_border)
                    # eval_result[metric+"_Y"] = self.allowed_metrics[metric](G_key_y, gt_y, crop_border)
                res.append(eval_result)
            return res
        else:
            return []
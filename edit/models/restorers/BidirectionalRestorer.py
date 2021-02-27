import os
import time
import numpy as np
import megengine.distributed as dist
import megengine as mge
import megengine.functional as F
from megengine.autodiff import GradManager
from edit.core.hook.evaluation import psnr, ssim
from edit.utils import imwrite, tensor2img, bgr2ycbcr, img_multi_padding, img_de_multi_padding, ensemble_forward, ensemble_back
from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS
from tqdm import tqdm

def get_bilinear(image):
    B,T,C,h,w = image.shape
    image = image.reshape(-1, C,h,w)
    return F.nn.interpolate(image, scale_factor=4).reshape(B,T,C,4*h, 4*w)

def train_generator_batch(image, label, *, gm, netG, netloss):
    B,T,_,h,w = image.shape
    _,_,_,H,W = label.shape
    biup = get_bilinear(image)
    netG.train()
    with gm:
        forward_hiddens = []
        backward_hiddens = []
        res = []
        # cal forward hiddens
        hidden = F.zeros((B, netG.hidden_channels, h, w))
        for i in range(T):
            now_frame = image[:, i, ...]
            if i==0:
                flow = netG.flownet(now_frame, now_frame)
                # print(F.max(flow), F.mean(flow))
            else:
                flow = netG.flownet(now_frame, image[:, i-1, ...])
                # print(F.max(flow), F.mean(flow))
            hidden = netG(hidden, flow, now_frame)
            forward_hiddens.append(hidden)
        # cal backward hiddens
        hidden = F.zeros((B, netG.hidden_channels, h, w))
        for i in range(T-1, -1, -1):
            now_frame = image[:, i, ...]
            if i==(T-1):
                flow = netG.flownet(now_frame, now_frame)
            else:
                flow = netG.flownet(now_frame, image[:, i+1, ...])
            hidden = netG(hidden, flow, now_frame)
            backward_hiddens.append(hidden)
        # do upsample for all frames
        for i in range(T):
            res.append(netG.do_upsample(forward_hiddens[i], backward_hiddens[T-i-1]))
        res = F.stack(res, axis = 1) # [B,T,3,H,W]
        loss = netloss(res+biup, label)  #  * 4*3*256*256  # same with official edvr   目标5.5
        gm.backward(loss)
        if dist.is_distributed():
            loss = dist.functional.all_reduce_sum(loss) / dist.get_world_size()
    return loss

def test_generator_batch(image, *, netG):
    # image: [1,100,3,180,320]
    # 要不要进行pad?
    B,T,_,h,w = image.shape
    biup = get_bilinear(image)
    netG.eval()
    forward_hiddens = []
    backward_hiddens = []
    res = []
    # cal forward hiddens
    hidden = F.zeros((B, netG.hidden_channels, h, w))
    for i in tqdm(range(T)):
        now_frame = image[:, i, ...]
        if i==0:
            flow = netG.flownet(now_frame, now_frame)
            # print(F.max(flow), F.mean(flow))
        else:
            flow = netG.flownet(now_frame, image[:, i-1, ...])
            # print(F.max(flow), F.mean(flow))
        hidden = netG(hidden, flow, now_frame)
        forward_hiddens.append(hidden)
    # cal backward hiddens
    hidden = F.zeros((B, netG.hidden_channels, h, w))
    for i in tqdm(range(T-1, -1, -1)):
        now_frame = image[:, i, ...]
        if i==(T-1):
            flow = netG.flownet(now_frame, now_frame)
        else:
            flow = netG.flownet(now_frame, image[:, i+1, ...])
        hidden = netG(hidden, flow, now_frame)
        backward_hiddens.append(hidden)
    # do upsample for all frames
    for i in tqdm(range(T)):
        res.append(netG.do_upsample(forward_hiddens[i], backward_hiddens[T-i-1]))
    res = F.stack(res, axis = 1) # [1,T,3,H,W]
    return res+biup

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    pass
    # for param_group in optimizer.param_groups:
    #     print(param_group['lr'])
    
@MODELS.register_module()
class BidirectionalRestorer(BaseModel):
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self, generator, pixel_loss, train_cfg=None, eval_cfg=None, pretrained=None):
        super(BidirectionalRestorer, self).__init__()

        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        # generator
        self.generator = build_backbone(generator)
        # loss
        self.pixel_loss = build_loss(pixel_loss)

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
        gt = batchdata['gt']  # [1,3,h,w]
        lq_paths = [item[0] for item in batchdata['lq_path']] # 3
        now_id = self.get_img_id(lq_paths[1]) # 1对应中间帧
        if now_id==0:
            print("first frame: {}".format(lq_paths[1]))
            self.LR_list = []
            self.HR_list = []
        
        self.LR_list.append(mge.tensor(lq[:, 1, ...], dtype="float32")) # [1,3,h,w]
        self.HR_list.append(gt) # numpy

        if now_id == 99:
            # 计算所有帧
            # stack所有LR
            print("start to forward and eval....")
            self.HR_G = test_generator_batch(F.stack(self.LR_list, axis=1), netG=self.generator)
        
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
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

def get_bilinear(image):
    B,T,C,h,w = image.shape
    image = image.reshape(-1, C,h,w)
    return F.nn.interpolate(image, scale_factor=4).reshape(B,T,C,4*h, 4*w)

def train_generator_batch(image, label, *, gm, netG1, netG2, netloss):
    B,T,_,h,w = image.shape
    biup = get_bilinear(image)
    netG1.eval()
    netG2.train()
    # get hidden of layer1
    forward_hiddens = []
    backward_hiddens = []
    res_layer1 = []
    hidden = F.zeros((2*B, netG1.hidden_channels, h, w))
    for i in range(T):
        now_frame = F.concat([image[:, i, ...], image[:, T-i-1, ...]], axis=0)
        if i==0:
            flow = netG1.flownet(now_frame, now_frame)
        else:
            ref = F.concat([image[:, i-1, ...], image[:, T-i, ...]], axis=0)
            flow = netG1.flownet(now_frame, ref)
        hidden = netG1(hidden, flow, now_frame)
        forward_hiddens.append(hidden[0:B, ...])
        backward_hiddens.append(hidden[B:2*B, ...])
    for i in range(T):
        res_layer1.append(netG1.prepare_for_layer_2(forward_hiddens[i], backward_hiddens[T-i-1]))
    # start to train layer2
    with gm:
        forward_hiddens = []
        backward_hiddens = []
        res = []
        hidden = F.zeros((2*B, netG2.hidden_channels, h, w))
        for i in range(T):
            now_frame = F.concat([image[:, i, ...], image[:, T-i-1, ...]], axis=0)
            if i==0:
                flow = netG2.flownet(now_frame, now_frame)
            else:
                ref = F.concat([image[:, i-1, ...], image[:, T-i, ...]], axis=0)
                flow = netG2.flownet(now_frame, ref)
            hidden = netG2(hidden, flow, now_frame, F.concat([res_layer1[i], res_layer1[T-i-1]], axis=0))
            forward_hiddens.append(hidden[0:B, ...])
            backward_hiddens.append(hidden[B:2*B, ...])
        for i in range(T):
            res.append(netG2.do_upsample(forward_hiddens[i], backward_hiddens[T-i-1], res_layer1[i]))
        res = F.stack(res, axis = 1) # [B,T,3,H,W]
        loss = netloss(res+biup, label)
        gm.backward(loss)
        if dist.is_distributed():
            loss = dist.functional.all_reduce_sum(loss) / dist.get_world_size()
    return loss

def test_generator_batch(image, *, netG1, netG2):
    B,T,_,h,w = image.shape
    biup = get_bilinear(image)
    netG1.eval()
    netG2.train()
    # get hidden of layer1
    forward_hiddens = []
    backward_hiddens = []
    res_layer1 = []
    hidden = F.zeros((2*B, netG1.hidden_channels, h, w))
    for i in range(T):
        now_frame = F.concat([image[:, i, ...], image[:, T-i-1, ...]], axis=0)
        if i==0:
            flow = netG1.flownet(now_frame, now_frame)
        else:
            ref = F.concat([image[:, i-1, ...], image[:, T-i, ...]], axis=0)
            flow = netG1.flownet(now_frame, ref)
        hidden = netG1(hidden, flow, now_frame)
        forward_hiddens.append(hidden[0:B, ...])
        backward_hiddens.append(hidden[B:2*B, ...])
    for i in range(T):
        res_layer1.append(netG1.prepare_for_layer_2(forward_hiddens[i], backward_hiddens[T-i-1]))
    #####################################################
    forward_hiddens = []
    backward_hiddens = []
    res = []
    hidden = F.zeros((2*B, netG2.hidden_channels, h, w))
    for i in range(T):
        now_frame = F.concat([image[:, i, ...], image[:, T-i-1, ...]], axis=0)
        if i==0:
            flow = netG2.flownet(now_frame, now_frame)
        else:
            ref = F.concat([image[:, i-1, ...], image[:, T-i, ...]], axis=0)
            flow = netG2.flownet(now_frame, ref)
        hidden = netG2(hidden, flow, now_frame, F.concat([res_layer1[i], res_layer1[T-i-1]], axis=0))
        forward_hiddens.append(hidden[0:B, ...])
        backward_hiddens.append(hidden[B:2*B, ...])
    for i in range(T):
        res.append(netG2.do_upsample(forward_hiddens[i], backward_hiddens[T-i-1], res_layer1[i]))
    res = F.stack(res, axis = 1) # [B,T,3,H,W]
    return res + biup

epoch_dict = {}

def adjust_learning_rate(optimizer, epoch):
    if epoch>=8 and epoch % 1 == 0  and epoch_dict.get(epoch, None) is None:
        epoch_dict[epoch] = True
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * 0.8
        print("adjust lr! , now lr: {}".format(param_group["lr"]))


@MODELS.register_module()
class BidirectionalRestorer_layer2(BaseModel):
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self, generator1, generator2, pixel_loss, train_cfg=None, eval_cfg=None, pretrained=None):
        super(BidirectionalRestorer_layer2, self).__init__()

        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        # generator
        self.generator1 = build_backbone(generator1)
        self.generator2 = build_backbone(generator2)
        # loss
        self.pixel_loss = build_loss(pixel_loss)
        # load pretrained
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        print("🤡 loading for layer1")
        state_dict = mge.load(self.generator2.pretrained_layer_1_path)
        self.generator1.load_state_dict(state_dict, strict=False)
        self.generator2.init_weights(pretrained)

    def train_step(self, batchdata, now_epoch):
        LR_tensor = mge.tensor(batchdata['lq'], dtype="float32")
        HR_tensor = mge.tensor(batchdata['gt'], dtype="float32")
        loss = train_generator_batch(LR_tensor, HR_tensor, gm=self.gms['generator2'], netG1=self.generator1, netG2=self.generator2, netloss=self.pixel_loss)
        adjust_learning_rate(self.optimizers['generator2'], now_epoch)
        self.optimizers['generator2'].step()
        self.optimizers['generator2'].clear_grad()
        return loss

    def get_img_id(self, key):
        assert isinstance(key, str)
        L = key.split("/")
        return int(L[-1][:-4]), L[-2] # id clip

    def test_step(self, batchdata, **kwargs):
        """
            possible kwargs:
                save_image
                save_path
                ensemble
        """
        lq = batchdata['lq']  #  [B,3,h,w]
        gt = batchdata.get('gt', None)  # if not None: [B,3,4*h,4*w]
        assert len(batchdata['lq_path']) == 1  # 每个sample所带的lq_path列表长度仅为1， 即自己
        lq_paths = batchdata['lq_path'][0] # length 为batch长度
        now_start_id, clip = self.get_img_id(lq_paths[0])
        now_end_id, _ = self.get_img_id(lq_paths[-1])
        assert clip == _
        if now_start_id==0:
            print("first frame: {}".format(lq_paths[0]))
            self.LR_list = []
            self.HR_list = []

        # pad lq
        B ,_ ,origin_H, origin_W = lq.shape
        lq = img_multi_padding(lq, padding_multi=self.eval_cfg.multi_pad, pad_method = "edge") #  edge  constant
        self.LR_list.append(mge.tensor(lq, dtype="float32"))  # [1,3,h,w]

        if gt is not None:
            for i in range(B):
                self.HR_list.append(gt[i:i+1, ...])

        if now_end_id == 99:
            print("start to forward all frames....")
            if self.eval_cfg.gap == 1:
                self.LR_list = F.concat(self.LR_list, axis=0) # [100, 3,h,w]
                self.HR_G = test_generator_batch(F.expand_dims(self.LR_list, axis=0), netG1=self.generator1, netG2=self.generator2)
            elif self.eval_cfg.gap == 2:
                raise NotImplementedError("not implement gap != 1 now")
                # self.HR_G_1 = test_generator_batch(F.stack(self.LR_list[::2], axis=1), netG=self.generator)
                # self.HR_G_2 = test_generator_batch(F.stack(self.LR_list[1::2], axis=1), netG=self.generator) # [B,T,C,H,W]
                # # 交叉组成HR_G
                # res = []
                # _,T1,_,_,_ = self.HR_G_1.shape
                # _,T2,_,_,_ = self.HR_G_2.shape
                # assert T1 == T2
                # for i in range(T1):
                #     res.append(self.HR_G_1[:, i, ...])
                #     res.append(self.HR_G_2[:, i, ...])
                # self.HR_G = F.stack(res, axis=1) # [B,T,C,H,W]
            else:
                raise NotImplementedError("do not support eval&test gap value")
            
            scale = self.generator2.upscale_factor
            # get numpy
            self.HR_G = img_de_multi_padding(self.HR_G.numpy(), origin_H=origin_H * scale, origin_W=origin_W * scale) # depad for HR_G   [B,T,C,H,W]

            if kwargs.get('save_image', False):
                print("saving images to disk ...")
                save_path = kwargs.get('save_path')
                B,T,_,_,_ = self.HR_G.shape
                assert B == 1
                assert T == 100
                for i in range(T):
                    if (i+1)%10 == 0:
                        imwrite(tensor2img(self.HR_G[0, i, ...], min_max=(0, 1)), file_path=os.path.join(save_path, f"{clip}_{str(i).zfill(8)}.png"))

        return now_end_id == 99

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
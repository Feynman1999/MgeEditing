import os
import time
import numpy as np
import megengine.distributed as dist
import megengine as mge
import megengine.functional as F
from collections import defaultdict
from megengine.autodiff import GradManager
from edit.core.hook.evaluation import psnr, ssim
from edit.utils import imwrite, tensor2img, bgr2ycbcr, img_multi_padding, img_de_multi_padding, ensemble_forward, ensemble_back
from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS

def train_generator_batch(image, label, trans_flags, *, gm, netG, netloss):
    """
        image: list of tensor, (1,3,h,w)
        label: list of tensor, (1,3,H,W)
    """
    T = len(image)
    biup = [ F.nn.interpolate(item, scale_factor=4) for item in image]
    netG.train()
    with gm:
        output = netG(image, trans_flags)
        for idx in range(T):
            output[idx] += biup[idx]
        loss = netloss(output, label)
        gm.backward(loss)
        if dist.is_distributed():
            loss = dist.functional.all_reduce_sum(loss) / dist.get_world_size()
    return loss

def test_generator_batch(image, *, netG):
    T = len(image)
    biup = [ F.nn.interpolate(item, scale_factor=4) for item in image]
    netG.eval()
    output = netG(image, [False])
    for idx in range(T):
        output[idx] += biup[idx]
    return output # list of tensor

@MODELS.register_module()
class EFCRestorer(BaseModel):
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self, generator, pixel_loss, train_cfg=None, eval_cfg=None, pretrained=None):
        super(EFCRestorer, self).__init__()

        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg

        # generator
        self.generator = build_backbone(generator)
        # loss
        self.pixel_loss = build_loss(pixel_loss)

        # load pretrained
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        self.generator.init_weights(pretrained)

    def train_step(self, batchdata, now_epoch, now_iter):
        """train step.

        Args:
            batchdata: list for train_batch, numpy.ndarray, length up to Collect class.
        Returns:
            list: loss
        """
        LR_tensor =  [mge.tensor(item, dtype="float32") for item in batchdata['lq']]
        HR_tensor =  [mge.tensor(item, dtype="float32") for item in batchdata['gt']]
        # for idx, item in enumerate(LR_tensor):
        #     imwrite(tensor2img(item[0]), file_path="./viz_efc/{}/LR_{}.png".format(self.xxx, idx))
        #     imwrite(tensor2img(HR_tensor[idx][0]), file_path="./viz_efc/{}/HR_{}.png".format(self.xxx, idx))
        # self.xxx += 1            
        # B,t,c,h,w = LR_tensor.shape
        # for i in range(B):
        #     for t in range(t):
        #         print(i, t)
        #         imwrite(tensor2img(LR_tensor[i,t], min_max=(-0.5, 0.5)), file_path="./haha/LR_{}_{}.png".format(i, t))
        #         imwrite(tensor2img(HR_tensor[i,t], min_max=(-0.5, 0.5)), file_path="./haha/HR_{}_{}.png".format(i, t))
        loss = train_generator_batch(LR_tensor, HR_tensor, batchdata['transpose'], gm=self.gms['generator'], netG=self.generator, netloss=self.pixel_loss)
        # for key,_ in self.generator.encoder.named_parameters():
        #     print(key)
        # print(self.generator.encoder[0].weight.grad)
        self.optimizers['generator'].step()
        self.optimizers['generator'].clear_grad()
        return loss

    def get_img_id(self, key):
        shift = self.eval_cfg.get('save_shift', 0)
        assert isinstance(key, str)
        L = key.split("/")
        return int(L[-1][:-4]), str(int(L[-2]) - shift).zfill(3) # id, clip

    def test_step_block_by_block(self, inp, ans):
        """
            ans: (1, 100, 3, 4*now_h, 4*now_w) ndarray
            inp: (100, 3, now_h, now_w) tensor
        """
        # 20帧一组，分4*4块处理，结果放到ans中
        T, _, now_h, now_w = inp.shape
        block_h = now_h // 2
        block_w = now_w // 2
        for seg in range(0,T,20):# seg: seg+20
            for i in range(2):
                for j in range(2):
                    print("seg: {}-{},  i:{},  j:{}".format(seg, seg+20-1, i, j))
                    inp_list = []
                    for t in range(seg, seg+20):
                        inp_list.append(inp[t:t+1, :, i*block_h:(i*block_h+block_h), j*block_w:(j*block_w+block_w)])
                    oup_list = test_generator_batch(inp_list, netG = self.generator)
                    # 可视化
                    # if i==0 and j==0:
                    #     imwrite(tensor2img(oup_list[0][0, ...], min_max=(-0.5, 0.5)), file_path="./xxx_seg_{}.png".format(seg))
                    # set ans
                    for t in range(20):
                        ans[0:1, seg + t, :, i*4*block_h:(i*4*block_h+4*block_h), j*4*block_w:(j*4*block_w+4*block_w)] = oup_list[t].numpy()

    def test_step(self, batchdata, **kwargs):
        """
            每一次传入帧i和光流i->i+1，最后一帧为99，没有光流，将这些信息储存起来
            如果当前帧是最后一帧，则开始处理程序：
            for i in range(0,99):
                如果当前帧还有块没有处理，则挖出来一条通道进行处理，将结果填入桶中，
                直到当前帧处理完毕。
                通道长度：min(T_train, 99-i+1)

            possible kwargs:
                save_image
                save_path
                ensemble
        
            可以测试不同的分块方式
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
        self.LR_list.append(lq)  # [1,3,h,w]

        if gt is not None:
            for i in range(B):
                self.HR_list.append(gt[i:i+1, ...])

        if now_end_id == 99:
            print("start to forward all frames....")
            if self.eval_cfg.gap == 1:
                # do ensemble (8 times) or not do (1 times)
                ensemble_res = []
                self.LR_list = np.concatenate(self.LR_list, axis=0) # [100, 3,h,w]
                for item in range(1): # do not have flip
                    print("do ensemble for {}".format(item))
                    inp = mge.tensor(ensemble_forward(self.LR_list, Type=item), dtype="float32")
                    _, _, now_h, now_w = inp.shape
                    self.HR_G = np.zeros((1, 100, 3, 4*now_h, 4*now_w)) # 往里面进行填充
                    self.test_step_block_by_block(inp, self.HR_G)
                    ensemble_res.append(ensemble_back(self.HR_G, Type=item))
                self.HR_G = sum(ensemble_res) / len(ensemble_res) # ensemble_res 结果取平均
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
            
            scale = self.generator.upscale_factor
            # get numpy
            self.HR_G = img_de_multi_padding(self.HR_G, origin_H=origin_H * scale, origin_W=origin_W * scale) # depad for HR_G   [B,T,C,H,W]

            if kwargs.get('save_image', False):
                print("saving images to disk ...")
                save_path = kwargs.get('save_path')
                B,T,_,_,_ = self.HR_G.shape
                assert B == 1
                assert T == 100
                for i in range(T):
                    img = tensor2img(self.HR_G[0, i, ...], min_max=(-0.5, 0.5))
                    if (i+1)%10 == 0:
                        imwrite(img, file_path=os.path.join(save_path, "partframes", f"{clip}_{str(i).zfill(8)}.png"))
                    # imwrite(img, file_path=os.path.join(save_path, "allframes", f"{clip}_{str(i).zfill(8)}.png"))
                    
        return now_end_id == 99

    def cal_for_eval(self, gathered_outputs, gathered_batchdata):
        """
            :param gathered_outputs: list of tensor, [B,C,H,W]
            :param gathered_batchdata: dict, include data
            :return: eval result
        """
        if gathered_outputs:
            crop_border = self.eval_cfg.crop_border
            assert len(self.HR_list) == 100
            res = []
            for i in range(len(self.HR_list)):
                G = tensor2img(self.HR_G[0, i, ...], min_max=(-0.5, 0.5))
                gt = tensor2img(self.HR_list[i][0], min_max=(-0.5, 0.5))
                eval_result = dict()
                for metric in self.eval_cfg.metrics:
                    eval_result[metric+"_RGB"] = self.allowed_metrics[metric](G, gt, crop_border)
                    # eval_result[metric+"_Y"] = self.allowed_metrics[metric](G_key_y, gt_y, crop_border)
                res.append(eval_result)
            return res
        else:
            return []

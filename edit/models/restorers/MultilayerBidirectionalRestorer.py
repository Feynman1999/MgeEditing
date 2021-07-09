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
from .MBR import shortcut_train_generator_batch, shortcut_test_generator_batch
from .MBR import viz_flow_train_generator_batch
from .MBR import flowmask_train_generator_batch, flowmask_test_generator_batch
from .MBR import baseline_test_generator_batch, baseline_train_generator_batch

epoch_dict = {}

def adjust_learning_rate(optimizer, epoch):
    if epoch>=8 and epoch % 1 == 0  and epoch_dict.get(epoch, None) is None:
        epoch_dict[epoch] = True
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * 0.9
        print("adjust lr! , now lr: {}".format(param_group["lr"]))

# TODO 可以再写一个父类，抽象一些公共方法，当大于1个模型时，代码重复了，如getimdid和test step
@MODELS.register_module()
class MultilayerBidirectionalRestorer(BaseModel):
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self, generator, pixel_loss, train_cfg=None, eval_cfg=None, pretrained=None, Fidelity_loss=None):
        super(MultilayerBidirectionalRestorer, self).__init__()

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

        img_norm_cfg = self.train_cfg.get("img_norm_cfg", None)
        assert img_norm_cfg, "train_cfg should contain key: img_norm_cfg"
        mean = img_norm_cfg.get("mean", None)
        if mean:
            self.min_max = (-mean[0], 1-mean[0])
        else:
            self.min_max = (0, 1)
        print("min max: {}".format(self.min_max))

        # set train and eval function
        train_cfg_dict = self.train_cfg.get("train_cfg", None)
        assert train_cfg_dict, "train_cfg should contain key: train_cfg"
        use_highway = train_cfg_dict.get("use_highway", None)
        use_flow_mask = train_cfg_dict.get("use_flow_mask", None)
        use_gap = train_cfg_dict.get("use_gap", None)
        viz_flow = train_cfg_dict.get("viz_flow", None)
        assert (use_highway is not None) and (use_flow_mask is not None) and (use_gap is not None)
        
        if use_gap:
            raise NotImplementedError("")
        elif use_flow_mask:
            print("use flow mask")
            self.train_generator_batch = flowmask_train_generator_batch
            self.test_generator_batch = flowmask_test_generator_batch
        elif use_highway:
            print("use highway")
            self.train_generator_batch = shortcut_train_generator_batch
            self.test_generator_batch = shortcut_test_generator_batch
        else:
            print("use baseline")
            self.train_generator_batch = baseline_train_generator_batch
            self.test_generator_batch = baseline_test_generator_batch

        if viz_flow:
            print("use viz flow")
            self.train_generator_batch = viz_flow_train_generator_batch

    def init_weights(self, pretrained=None):
        self.generator.init_weights(pretrained)

    def train_step(self, batchdata, now_epoch, now_iter):
        LR_tensor = mge.tensor(batchdata['lq'], dtype="float32")
        HR_tensor = mge.tensor(batchdata['gt'], dtype="float32")
        loss = self.train_generator_batch(LR_tensor, HR_tensor, gm=self.gms['generator'], netG=self.generator, netloss=self.pixel_loss)
        adjust_learning_rate(self.optimizers['generator'], now_epoch)
        self.optimizers['generator'].step()
        self.optimizers['generator'].clear_grad()
        return loss

    def get_img_id(self, key):
        shift = self.eval_cfg.get('save_shift', 0)
        assert isinstance(key, str)
        L = key.split("/")
        return int(L[-1][:-4]), str(int(L[-2]) - shift).zfill(3) # id, clip

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
        self.LR_list.append(lq)  # [1,3,h,w]

        if gt is not None:
            for i in range(B):
                self.HR_list.append(gt[i:i+1, ...])

        if now_end_id == 99:
            print("start to forward all frames....")
            if self.eval_cfg.gap == 1:
                # do ensemble (8 times)
                ensemble_res = []
                self.LR_list = np.concatenate(self.LR_list, axis=0) # [100, 3,h,w]
                for item in range(1): # do not have flip
                    inp = mge.tensor(ensemble_forward(self.LR_list, Type=item), dtype="float32")
                    oup = self.test_generator_batch(F.expand_dims(inp, axis=0), netG=self.generator)
                    ensemble_res.append(ensemble_back(oup.numpy(), Type=item))
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
                    img = tensor2img(self.HR_G[0, i, ...], min_max=self.min_max)
                    if (i+1)%10 == 0:
                        imwrite(img, file_path=os.path.join(save_path, "partframes", f"{clip}_{str(i).zfill(8)}.png"))
                    imwrite(img, file_path=os.path.join(save_path, "allframes", f"{clip}_{str(i).zfill(8)}.png"))
                    
        return now_end_id == 99

    def cal_for_eval(self, gathered_outputs, gathered_batchdata):
        if gathered_outputs:
            crop_border = self.eval_cfg.crop_border
            assert len(self.HR_list) == 100
            res = []
            for i in range(len(self.HR_list)):
                G = tensor2img(self.HR_G[0, i, ...], min_max=self.min_max)
                gt = tensor2img(self.HR_list[i][0], min_max=self.min_max)
                eval_result = dict()
                for metric in self.eval_cfg.metrics:
                    eval_result[metric+"_RGB"] = self.allowed_metrics[metric](G, gt, crop_border)
                    # eval_result[metric+"_Y"] = self.allowed_metrics[metric](G_key_y, gt_y, crop_border)
                res.append(eval_result)
            return res
        else:
            return []
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

def train_generator_batch(image, *, gm, netG, netloss):
    pass

def test_generator_batch(image, *, netG):
    pass

epoch_dict = {}

def adjust_learning_rate(optimizer, epoch):
    if epoch>=20 and epoch % 1 == 0  and epoch_dict.get(epoch, None) is None:
        epoch_dict[epoch] = True
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * 0.8
        print("adjust lr! , now lr: {}".format(param_group["lr"]))

def get_bilinear(image):
    B,T,C,h,w = image.shape
    image = image.reshape(-1, C,h,w)
    return F.nn.interpolate(image, scale_factor=4).reshape(B,T,C,4*h, 4*w)

@MODELS.register_module()
class MeshFlowMatching(BaseModel):
    allowed_metrics = {'PSNR': psnr}

    def __init__(self, generator, pixel_loss, train_cfg=None, eval_cfg=None, pretrained=None):
        super(MeshFlowMatching, self).__init__()

        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        self.generator = build_backbone(generator)
        self.pixel_loss = build_loss(pixel_loss)
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        self.generator.init_weights(pretrained)

    def train_step(self, batchdata, now_epoch, now_iter):
        LR_tensor = mge.tensor(batchdata['lq'], dtype="float32")
        # print(LR_tensor.shape) # (8, 3, 3, 54, 96)
        # B,T,_,_,_ = viz.shape
        # for i in range(T):
        #     imwrite(tensor2img(viz[0, i, ...]), file_path="./workdirs/visual/iter_{}_{}.png".format(now_iter, i))
        loss = train_generator_batch(LR_tensor, gm=self.gms['generator'], netG=self.generator, netloss=self.pixel_loss)
        adjust_learning_rate(self.optimizers['generator'], now_epoch)
        self.optimizers['generator'].step()
        self.optimizers['generator'].clear_grad()
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
        if kwargs.get('save_image', False):
            print("saving images to disk ...")
            save_path = kwargs.get('save_path')
            pass

    def cal_for_eval(self, gathered_outputs, gathered_batchdata):
        pass
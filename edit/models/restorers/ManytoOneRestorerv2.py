import os
import time
from PIL import Image
import numpy as np
from megengine.jit import trace, SublinearMemoryConfig
import megengine.distributed as dist
import megengine as mge
from edit.core.hook.evaluation import psnr, ssim
from edit.utils import imwrite, tensor2img, bgr2ycbcr, img_multi_padding, img_de_multi_padding, ensemble_back, ensemble_forward, imrescale
from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS


def get_mid_bicubic(image, mid_idx = 3):
    mid_bicubic = image[:, mid_idx, ...] # [B,C,H,W]
    B, _, H, W  = mid_bicubic.shape
    res = []
    for i in range(B):
        out = imrescale(np.transpose(mid_bicubic[i, ...], (1,2,0)), scale = 4)
        res.append(np.transpose(out, (2, 0, 1)))
    return np.stack(res, axis= 0)  # [B,C,4*H,4*W]


def train_generator_batch(image, mid_bicubic, label, *, gm, netG, netloss):
    B,T,C,h,w = image.shape
    netG.train()
    with gm:
        output = netG(image, mid_bicubic)
        loss = netloss(output, label) / ( (h/64) * (w/64) * (B/4) ) # same with official edvr   4*3*256*256
        gm.backward(loss)
        if dist.is_distributed():
            loss = dist.functional.all_reduce_sum(loss) / dist.get_world_size()
    return loss


def test_generator_batch(image, mid_bicubic, *, netG):
    netG.eval()
    output = netG(image, mid_bicubic)
    return output


@MODELS.register_module()
class ManytoOneRestorer_v2(BaseModel):
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self, generator, pixel_loss, train_cfg=None, eval_cfg=None, pretrained=None):
        super(ManytoOneRestorer_v2, self).__init__()

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

    def train_step(self, batchdata):
        """train step.

        Args:
            batchdata: list for train_batch, numpy.ndarray, length up to Collect class.
        Returns:
            list: loss
        """
        data, label = batchdata['lq'], batchdata['gt']
        mid_bicubic = get_mid_bicubic(data)
        LR_tensor = mge.tensor(data, dtype="float32")
        HR_tensor = mge.tensor(label, dtype="float32")
        mid_bicubic = mge.tensor(mid_bicubic, dtype="float32")
        loss = train_generator_batch(LR_tensor, mid_bicubic, HR_tensor, gm=self.gms['generator'], netG=self.generator, netloss=self.pixel_loss)
        self.optimizers['generator'].step()
        self.optimizers['generator'].clear_grad()
        return loss

    def get_img_id(self, key):
        assert isinstance(key, str)
        return int(key.split("/")[-1][:-4])

    def test_step(self, batchdata, **kwargs):
        start_time = time.time()
        lq = batchdata['lq']
        lq_paths = [item[0] for item in batchdata['lq_path']]
        num_input_frames =  batchdata['num_input_frames'][0] # 3

        # get ids
        ids = [ self.get_img_id(path) for path in lq_paths]

        B,T,_,h,w = lq.shape
        assert B == 1, "only support batchsize==1 for test and eval now"

        lq_tensor = mge.tensor(lq, dtype="float32") # [B,T,C,H,W]
        mid_bicubic = get_mid_bicubic(lq)
        mid_bicubic = mge.tensor(mid_bicubic, dtype="float32")
        output = test_generator_batch(lq_tensor, mid_bicubic, netG = self.generator)  # HR [B,C,4H,4W]

        if kwargs.get('save_image'):
            pass

        print("imgs {} ok  inference time: {} s".format(ids, time.time() - start_time))
        return [output, ]

    def cal_for_eval(self, gathered_outputs, gathered_batchdata):
        crop_border = self.eval_cfg.crop_border
        gathered_outputs = gathered_outputs[0]
        gathered_batchdata = gathered_batchdata['gt']
        assert list(gathered_batchdata.shape) == list(gathered_outputs.shape), "{} != {}".format(list(gathered_batchdata.shape), list(gathered_outputs.shape))

        res = []
        sample_nums = gathered_outputs.shape[0]
        for i in range(sample_nums):
            output = tensor2img(gathered_outputs[i])
            output_y = bgr2ycbcr(output, y_only=True)
            gt = tensor2img(gathered_batchdata[i])
            gt_y = bgr2ycbcr(gt, y_only=True)
            eval_result = dict()
            for metric in self.eval_cfg.metrics:
                eval_result[metric+"_RGB"] = self.allowed_metrics[metric](output, gt, crop_border)
                eval_result[metric+"_Y"] = self.allowed_metrics[metric](output_y, gt_y, crop_border)
            res.append(eval_result)
        return res

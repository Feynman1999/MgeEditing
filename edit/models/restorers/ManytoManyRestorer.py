import os
import time
import numpy as np
from megengine.jit import trace
import megengine.distributed as dist
import megengine as mge
import megengine.functional as F
from edit.core.evaluation import psnr, ssim
from edit.utils import imwrite, tensor2img, bgr2ycbcr, img_multi_padding, img_de_multi_padding, ensemble_forward, ensemble_back
from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS

hidden_channels = 64

@trace(symbolic=True)
def train_generator_batch(image, label, *, opt, netG, netloss):
    netG.train()
    B,T,_,H,W = image.shape
    # image
    image_S = image.reshape((B*T, -1, H, W))
    image_S = F.interpolate(image_S, scale_factor = [0.25, 0.25])
    image_S = F.interpolate(image_S, size = [H, W])
    image_S = image_S.reshape((B, T, -1, H, W))
    image_D = image - image_S
    # label
    label_S = label.reshape((B*T, -1, 4*H, 4*W))
    label_S = F.interpolate(label_S, scale_factor = [0.25, 0.25])
    label_S = F.interpolate(label_S, size = [4 * H, 4 * W])
    label_S = label_S.reshape((B, T, -1, 4*H, 4*W))
    label_D = label - label_S

    HR_G = []
    HR_D = []
    HR_S = []

    pre_S_hat = mge.tensor(np.zeros((B, hidden_channels, H, W), dtype=np.float32))
    pre_D_hat = F.zeros_like(pre_S_hat)
    pre_SD = F.zeros_like(pre_S_hat)
    
    imgHR, pre_SD, pre_S_hat, pre_D_hat, img_S, img_D = netG(image[:, 0, ...], image_S[:, 0, ...], 
                                        image_D[:, 0, ...], image_S[:, 1, ...],
                                        image_D[:, 1, ...], pre_S_hat, pre_D_hat, pre_SD)
    HR_G.append(F.add_axis(imgHR, axis = 1))
    HR_D.append(F.add_axis(img_D, axis = 1))
    HR_S.append(F.add_axis(img_S, axis = 1))
    for t in range(1, T):
        imgHR, pre_SD, pre_S_hat, pre_D_hat, img_S, img_D = netG(image[:, t, ...], image_S[:, t, ...], 
                                    image_D[:, t, ...], image_S[:, t-1, ...],
                                    image_D[:, t-1, ...], pre_S_hat, pre_D_hat, pre_SD)
        HR_G.append(F.add_axis(imgHR, axis = 1))
        HR_D.append(F.add_axis(img_S, axis = 1))
        HR_S.append(F.add_axis(img_D, axis = 1))

    HR_G = F.concat(HR_G, axis = 1)
    HR_D = F.concat(HR_D, axis = 1)
    HR_S = F.concat(HR_S, axis = 1)
    # assert HR_G.shape == HR_D.shape and HR_D.shape == HR_S.shape # [B,T,C,H,W]
    loss = netloss(HR_G, HR_D, HR_S, label, label_D, label_S)
    opt.backward(loss)
    if dist.is_distributed():
        # do all reduce mean
        pass
    return loss


@trace(symbolic=True)
def test_generator_batch(image, pre_S, pre_D, pre_S_hat, pre_D_hat, pre_SD, *, netG):
    netG.eval()
    _,_,H,W = image.shape
    image_S = F.interpolate(image, scale_factor = [0.25, 0.25])
    image_S = F.interpolate(image_S, size = [H, W])
    image_D = image - image_S
    outputs = netG(image, image_S, image_D, pre_S, pre_D, pre_S_hat, pre_D_hat, pre_SD)
    return list(outputs)[:4] + [image_S, image_D]


@MODELS.register_module()
class ManytoManyRestorer(BaseModel):
    """ManytoManyRestorer for video restoration.

    It must contain a generator that takes some component as inputs and outputs 
    HR image.

    The subclasses should overwrite the function `test_step` and `train_step` and `cal_for_eval`.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        eval_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self, generator, pixel_loss, train_cfg=None, eval_cfg=None, pretrained=None):
        super(ManytoManyRestorer, self).__init__()
        global hidden_channels
        hidden_channels = generator.hidden_channels

        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg

        # generator
        self.generator = build_backbone(generator)
        # loss
        self.pixel_loss = build_loss(pixel_loss)

        # load pretrained
        self.init_weights(pretrained)

        self.now_test_num = 1

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
        data, label = batchdata
        self.optimizers['generator'].zero_grad()
        loss = train_generator_batch(data, label, opt=self.optimizers['generator'], netG=self.generator, netloss=self.pixel_loss)
        self.optimizers['generator'].step()
        return loss

    def test_step(self, batchdata, **kwargs):
        """test step.
           need to know whether the first frame for video, and every step restore some hidden state.
        Args:
            batchdata: list for train_batch, numpy.ndarray, length up to Collect class.

        Returns:
            list: outputs
        """
        epoch = kwargs.get('epoch', 0)
        image = batchdata[0]  # [B,C,H,W]
        image = ensemble_forward(image, Type = epoch)  # for ensemble

        H,W = image.shape[-2], image.shape[-1]
        scale = getattr(self.generator, 'upscale_factor', 4)
        padding_multi = self.eval_cfg.get('padding_multi', 1)
        # padding for H and W
        image = img_multi_padding(image, padding_multi = padding_multi, pad_value = -0.5)  # [B,C,H,W]
        
        assert image.shape[0] == 1  # only support batchsize 1
        assert len(batchdata[1].shape) == 1  # first frame flag
        if batchdata[1][0] > 0.5:  # first frame
            print("first frame")
            self.now_test_num = 1
            B, _ , now_H ,now_W = image.shape
            print("use now_H : {} and now_W: {}".format(now_H, now_W))
            self.pre_S_hat = np.zeros((B, hidden_channels, now_H, now_W), dtype=np.float32)
            self.pre_D_hat = np.zeros_like(self.pre_S_hat)
            self.pre_SD = np.zeros_like(self.pre_S_hat)
            self.pre_S = F.interpolate(mge.tensor(image), scale_factor = [0.25, 0.25])
            self.pre_S = F.interpolate(self.pre_S, size = [now_H, now_W]).numpy()
            self.pre_D = image - self.pre_S

        outputs = test_generator_batch(image, self.pre_S, self.pre_D, self.pre_S_hat, self.pre_D_hat, self.pre_SD, netG = self.generator)
        outputs = list(outputs)
        outputs[0] = img_de_multi_padding(outputs[0], origin_H = H*scale, origin_W = W*scale)
        
        for i in range(0, 6):
            outputs[i] = outputs[i].numpy()

        # update hidden state
        G, self.pre_SD, self.pre_S_hat, self.pre_D_hat, self.pre_S, self.pre_D = outputs
        
        # back ensemble for G
        G = ensemble_back(G, Type = epoch)

        save_image_flag = kwargs.get('save_image')
        if save_image_flag:
            save_path = kwargs.get('save_path', None)
            start_id = kwargs.get('sample_id', None)
            if save_path is None or start_id is None:
                raise RuntimeError("if save image in test_step, please set 'save_path' and 'sample_id' parameters")
            for idx in range(G.shape[0]):
                if epoch == 0:
                    imwrite(tensor2img(G[idx], min_max=(-0.5, 0.5)), file_path=os.path.join(save_path, "idx_{}.png".format(start_id + idx)))
                else:
                    imwrite(tensor2img(G[idx], min_max=(-0.5, 0.5)), file_path=os.path.join(save_path, "idx_{}_epoch_{}.png".format(start_id + idx, epoch)))

        print("now test num: {}".format(self.now_test_num))
        self.now_test_num += 1
        return outputs

    def cal_for_eval(self, gathered_outputs, gathered_batchdata):
        """

        :param gathered_outputs: list of tensor, [B,C,H,W]
        :param gathered_batchdata: list of numpy, [B,C,H,W]
        :return: eval result
        """
        crop_border = self.eval_cfg.crop_border
        gathered_outputs = gathered_outputs[0]  # image 
        gathered_batchdata = gathered_batchdata[-1]  # label
        assert list(gathered_batchdata.shape) == list(gathered_outputs.shape), "{} != {}".format(list(gathered_batchdata.shape), list(gathered_outputs.shape))

        res = []
        sample_nums = gathered_outputs.shape[0]
        for i in range(sample_nums):
            output = tensor2img(gathered_outputs[i], min_max=(-0.5, 0.5))
            output_y = bgr2ycbcr(output, y_only=True)
            gt = tensor2img(gathered_batchdata[i], min_max=(-0.5, 0.5))
            gt_y = bgr2ycbcr(gt, y_only=True)
            eval_result = dict()
            for metric in self.eval_cfg.metrics:
                eval_result[metric+"_RGB"] = self.allowed_metrics[metric](output, gt, crop_border)
                eval_result[metric+"_Y"] = self.allowed_metrics[metric](output_y, gt_y, crop_border)
            res.append(eval_result)
        return res

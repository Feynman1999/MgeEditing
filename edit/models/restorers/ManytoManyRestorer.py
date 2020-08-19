import os
import time
from megengine.jit import trace
import megengine.distributed as dist
import megengine as mge
from edit.core.evaluation import psnr, ssim
from edit.utils import imwrite, tensor2img, bgr2ycbcr
from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS


# @trace(symbolic=True)
def train_generator_batch(image, label, *, opt, netG, netloss):
    netG.train()
    B,T,_,H,W = image.shape
    # cal S
    image_S = mge.functional.reshape(image, (B*T, -1, H, W))
    image_S = mge.functional.interpolate(image_S, scale_factor = [0.25, 0.25])
    image_S = mge.functional.interpolate(image_S, size = [H, W])
    image_S = mge.functional.reshape(image_S, (B, T, -1, H, W))
    image_D = image - image_S
    # cal D
    label_S = mge.functional.reshape(label, (B*T, -1, 4*H, 4*W))
    label_S = mge.functional.interpolate(label_S, scale_factor = [0.25, 0.25])
    label_S = mge.functional.interpolate(label_S, size = [4 * H, 4 * W])
    label_S = mge.functional.reshape(label_S, (B, T, -1, 4*H, 4*W))
    label_D = label - label_S

    HR_G = []
    HR_D = []
    HR_S = []
    for t in range(T):
        if t>0:
            imgHR, SD, S_hat, D_hat, img_S, img_D = netG(image[:, t, ...], image_S[:, t, ...], 
                                        image_D[:, t, ...], image_S[:, t-1, ...],
                                        image_D[:, t-1, ...], S_hat, D_hat, SD)
        else:
            imgHR, SD, S_hat, D_hat, img_S, img_D = netG(image[:, 0, ...], image_S[:, 0, ...], 
                                        image_D[:, 0, ...], image_S[:, 1, ...],
                                        image_D[:, 1, ...])
        HR_G.append(mge.functional.add_axis(imgHR, axis = 1))
        HR_D.append(mge.functional.add_axis(img_S, axis = 1))
        HR_S.append(mge.functional.add_axis(img_D, axis = 1))

    HR_G = mge.functional.concat(HR_G, axis = 1)
    HR_D = mge.functional.concat(HR_D, axis = 1)
    HR_S = mge.functional.concat(HR_S, axis = 1)
    assert HR_G.shape == HR_D.shape and HR_D.shape == HR_S.shape # [B,T,C, H,W]
    loss = netloss(HR_G, HR_D, HR_S, label, label_D, label_S)
    opt.backward(loss)
    if dist.is_distributed():
        # do all reduce mean
        pass
    return loss


@trace(symbolic=True)
def test_generator_batch(image, *, netG):
    netG.eval()
    output = netG(image)
    return output


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

        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg

        # generator
        self.generator = build_backbone(generator)
        # loss
        self.pixel_loss = build_loss(pixel_loss)

        # load pretrained
        self.init_weights(pretrained)

        self.data = mge.tensor()
        self.label = mge.tensor()

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
        self.data.set_value(data)  # (4, 7, 3, 64, 64)
        self.label.set_value(label)  # (4, 7, 3, 256, 256)
        self.optimizers['generator'].zero_grad()
        loss = train_generator_batch(self.data, self.label, opt=self.optimizers['generator'], netG=self.generator, netloss=self.pixel_loss)
        self.optimizers['generator'].step()  # 根据梯度更新参数值
        return loss

    def test_step(self, batchdata, **kwargs):
        """test step.

        Args:
            batchdata: list for train_batch, numpy.ndarray or variable, length up to Collect class.

        Returns:
            list: outputs (already gathered from all threads)
        """
        output = test_generator_batch(batchdata[0], netG = self.generator)
        save_image_flag = kwargs.get('save_image')
        if save_image_flag:
            save_path = kwargs.get('save_path', None)
            start_id = kwargs.get('sample_id', None)
            if save_path is None or start_id is None:
                raise RuntimeError("if save image in test_step, please set 'save_path' and 'sample_id' parameters")
            G = output
            for idx in range(G.shape[0]):
                imwrite(tensor2img(G[idx], min_max=(-0.5, 0.5)), file_path=save_path + "_idx_{}.png".format(start_id + idx))
        return [output, ]

    def cal_for_eval(self, gathered_outputs, gathered_batchdata):
        """

        :param gathered_outputs: list of numpy, [B,C,H,W]
        :param gathered_batchdata: list of variable, [B,C,H,W]
        :return: eval result
        """
        crop_border = self.eval_cfg.crop_border
        gathered_outputs = gathered_outputs[0]
        gathered_batchdata = gathered_batchdata[-1]
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

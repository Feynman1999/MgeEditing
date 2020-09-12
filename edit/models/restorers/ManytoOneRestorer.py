import os
import time
from megengine.jit import trace, SublinearMemoryConfig
import megengine.distributed as dist
import megengine as mge
from edit.core.evaluation import psnr, ssim
from edit.utils import imwrite, tensor2img, bgr2ycbcr, img_multi_padding, img_de_multi_padding, ensemble_back, ensemble_forward
from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS

config = SublinearMemoryConfig()

@trace(symbolic=True)
def train_generator_batch(image, label, *, opt, netG, netloss):
    netG.train()
    output = netG(image)
    loss = netloss(output, label)
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
class ManytoOneRestorer(BaseModel):
    """ManytoOneRestorer for video restoration.

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
        super(ManytoOneRestorer, self).__init__()

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
        self.optimizers['generator'].zero_grad()
        loss = train_generator_batch(data, label, opt=self.optimizers['generator'], netG=self.generator, netloss=self.pixel_loss)
        self.optimizers['generator'].step()  # 根据梯度更新参数值
        return loss

    def test_step(self, batchdata, **kwargs):
        """test step.

        Args:
            batchdata: list for train_batch, numpy.ndarray or variable, length up to Collect class.

        Returns:
            list: outputs (already gathered from all threads)
        """
        epoch = kwargs.get('epoch', 0)
        images = batchdata[0] # [B,N,C,H,W]
        images = ensemble_forward(images, Type = epoch)  # for ensemble

        H,W = images.shape[-2], images.shape[-1]
        scale = getattr(self.generator, 'upscale_factor', 4)
        padding_multi = self.eval_cfg.get('padding_multi', 1)
        # padding for H and W
        images = img_multi_padding(images, padding_multi = padding_multi, pad_value = -0.5)  # [B,N,C,H,W]
        output = test_generator_batch(images, netG = self.generator)  # HR [B,C,4H,4W]
        output = img_de_multi_padding(output, origin_H = H*scale, origin_W = W*scale)
        
        # back ensemble for G
        G = ensemble_back(output, Type = epoch)

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

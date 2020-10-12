import os
import time
import numpy as np
import random
import cv2
from megengine.jit import trace, SublinearMemoryConfig
import megengine.distributed as dist
import megengine as mge
import megengine.functional as F
from edit.utils import imwrite, tensor2img, bgr2ycbcr, imrescale
from ..base import BaseModel
from ..builder import build_backbone
from ..registry import MODELS

def get_box(xy_ctr, offsets):
    """
        xy_ctr: [1,2,19,19]
        offsets: [B,2,19,19]
    """
    
    xy0 = (xy_ctr - F.zeros_like(offsets) - 127.5)  # top-left
    xy1 = xy0 + 255  # bottom-right
    bboxes_pred = F.concat([xy0, xy1], axis=1)  # (B,4,H,W)
    return bboxes_pred

config = SublinearMemoryConfig()

@trace(symbolic=True)
def train_generator_batch(optical, sar, label, *, opt, netG):
    netG.train()
    cls_score, offsets, ctr_score = netG(sar, optical)
    loss = netG.loss(cls_score, offsets, ctr_score, label)
    opt.backward(loss)
    if dist.is_distributed():
        # do all reduce mean
        pass

    # performance in the training data
    B, _, _, _ = cls_score.shape
    cls_score = cls_score.reshape(B, -1)
    # find the max
    max_id = F.argmax(cls_score, axis = 1)  # (B, )
    pred_box = get_box(netG.fm_ctr, offsets)  # (B,4,H,W)
    pred_box = pred_box.reshape(B, 4, -1)
    output = []
    for i in range(B):
        output.append(F.add_axis(pred_box[i, :, max_id[i]], axis=0)) # (1, 4)
    output = F.concat(output, axis=0)  # (B, 4)
    return [loss, F.norm(output[:, 0:2] - label[:, 0:2], p=2, axis = 1).mean()]


@trace(symbolic=True)
def test_generator_batch(optical, sar, *, netG):
    netG.eval()
    cls_score, offsets, ctr_score = netG(sar, optical)  # [B,1,19,19]  [B,2,19,19]  [B,1,19,19]
    B, _, _, _ = cls_score.shape
    # 加权
    cls_score = cls_score * ctr_score
    cls_score = cls_score.reshape(B, -1)
    # find the max
    max_id = F.argmax(cls_score, axis = 1)  # (B, )
    pred_box = get_box(netG.fm_ctr, offsets)  # (B,4,H,W)
    pred_box = pred_box.reshape(B, 4, -1)
    output = []
    for i in range(B):
        output.append(F.add_axis(pred_box[i, :, max_id[i]], axis=0)) # (1, 4)
    return F.concat(output, axis=0)  # [B,4]

def eval_distance(pred, gt):  # (4, )
    assert len(pred.shape) == 1
    return np.linalg.norm(pred[0:2]-gt[0:2], ord=2)

@MODELS.register_module()
class BasicMatching(BaseModel):
    allowed_metrics = {'dis': eval_distance}

    def __init__(self, generator, train_cfg=None, eval_cfg=None, pretrained=None):
        super(BasicMatching, self).__init__()

        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg

        # generator
        self.generator = build_backbone(generator)

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
        optical, sar, label = batchdata
        # 保存optical 和 sar，看下对不对
        # name = random.sample('zyxwvutsrqponmlkjihgfedcba', 3)
        # name = "".join(name) + "_" + str(label[0][0]) + "_" + str(label[0][1]) + "_" + str(label[0][2]) + "_" + str(label[0][3])
        # imwrite(cv2.rectangle(tensor2img(optical[0, ...], min_max=(-0.64, 1.36)), (label[0][1], label[0][0]), (label[0][3], label[0][2]), (0,0,255), 2), file_path="./workdirs/" + name + "_opt.png") 
        # imwrite(tensor2img(sar[0, ...], min_max=(-0.64, 1.36)), file_path="./workdirs/" + name + "_sar.png")
        self.optimizers['generator'].zero_grad()
        loss = train_generator_batch(optical, sar, label, opt=self.optimizers['generator'], netG=self.generator)
        self.optimizers['generator'].step()
        return loss

    def test_step(self, batchdata, **kwargs):
        """test step.

        Args:
            batchdata: list for train_batch, numpy.ndarray or variable, length up to Collect class.

        Returns:
            list: outputs (already gathered from all threads)
        """
        optical = batchdata[0]  # [B ,1 , H, W]
        sar = batchdata[1]
        
        pre_bbox = test_generator_batch(optical, sar, netG=self.generator)  # [B, 4]

        save_image_flag = kwargs.get('save_image')
        if save_image_flag:
            save_path = kwargs.get('save_path', None)
            start_id = kwargs.get('sample_id', None)
            if save_path is None or start_id is None:
                raise RuntimeError("if save image in test_step, please set 'save_path' and 'sample_id' parameters")
            for idx in range(pre_bbox.shape[0]):
                imwrite(tensor2img(optical[idx], min_max=(-0.64, 1.36)), file_path=os.path.join(save_path, "idx_{}.png".format(start_id + idx)))
                
        return [pre_bbox, ]

    def cal_for_eval(self, gathered_outputs, gathered_batchdata):
        """

        :param gathered_outputs: list of variable, [pre_bbox, ]
        :param gathered_batchdata: list of numpy, [optical, sar, bbox_gt]
        :return: eval result
        """
        gathered_outputs = gathered_outputs[0]
        gathered_batchdata = gathered_batchdata[-1]
        assert list(gathered_batchdata.shape) == list(gathered_outputs.shape), "{} != {}".format(list(gathered_batchdata.shape), list(gathered_outputs.shape))

        res = []
        sample_nums = gathered_outputs.shape[0]
        for i in range(sample_nums):
            eval_result = dict()
            for metric in self.eval_cfg.metrics:
                eval_result[metric+"_RGB"] = self.train_cfg['scale'] * self.allowed_metrics[metric](gathered_outputs[i].numpy(), gathered_batchdata[i])
            res.append(eval_result)
        return res

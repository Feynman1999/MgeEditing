import os
import time
import numpy as np
import random
import cv2
from megengine.jit import trace, SublinearMemoryConfig
import megengine.distributed as dist
import megengine as mge
import megengine.functional as F
from edit.utils import imwrite, tensor2img, bgr2ycbcr, imrescale, ensemble_forward, bbox_ensemble_back
from ..base import BaseModel
from ..builder import build_backbone
from ..registry import MODELS

config = SublinearMemoryConfig()


def add_H_W_Padding(x, margin=1):
    shape = x.shape
    padding_shape = list(shape)[:-2] + [ shape[-2] + 2*margin, shape[-1] + 2*margin ]
    res = mge.zeros(padding_shape, dtype=x.dtype)
    res = res.set_subtensor(x)[:, :, margin:margin + shape[-2],  margin: margin + shape[-1]]
    return res

@trace(symbolic=True)
def train_generator_batch(optical, sar, label, *, opt, netG):
    netG.train()
    cls_score = netG(sar, optical)
    loss = netG.loss(cls_score, label)
    opt.backward(loss)
    if dist.is_distributed():
        # do all reduce mean
        pass

    # performance in the training data
    B, _, H, W = cls_score.shape

    # find the max
    max_id = F.argmax(cls_score.reshape(B, -1), axis = 1)  # (B, )
    pred_box = netG.fm_ctr  # (1, 2, 181, 181)
    # cls_score = add_H_W_Padding(cls_score, margin = margin)

    output = []
    for i in range(B):
        H_id = max_id[i] // H
        W_id = max_id[i] % H
        # name = "".join(random.sample('zyxwvutsrqponmlkjihgfedcba',10))
        # tt = cls_score[i, 0, :, :]
        # # label周围3*3标0
        # label_H = int(label[i][0].item())
        # label_W = int(label[i][1].item())
        # img = tensor2img(tt, out_type=np.uint8, min_max=(F.min(tt).item(), F.max(tt).item()))
        # img = img[:, :, np.newaxis]
        # img[label_H-1:label_H+2, label_W-1:label_W+2, :] = 0
        # imwrite(img=img, file_path = "./test4/"+name+"_{}.png".format(i))
        output.append(F.add_axis(pred_box[0, :, H_id, W_id]-(netG.z_size-1)/2, axis=0)) # (1, 2)
    output = F.concat(output, axis=0)  # (B, 2)

    return [loss*1000, F.norm(output[:, 0:2] - label[:, 0:2], p=2, axis = 1).mean()]


@trace(symbolic=True)
def test_generator_batch(optical, sar, *, netG):
    netG.eval()
    tmp = netG.z_size
    netG.z_size = netG.test_z_size
    cls_score = netG(sar, optical)  # [B,1,19,19]  [B,2,19,19]  [B,1,19,19]
    B, _, H, W = cls_score.shape
    max_id = F.argmax(cls_score.reshape(B, -1), axis = 1)  # (B, )
    pred_box = netG.test_fm_ctr

    output = []
    for i in range(B):
        H_id = max_id[i] // H
        W_id = max_id[i] % H
        output.append(F.add_axis(pred_box[0, :, H_id, W_id]-(netG.z_size-1)/2, axis=0)) # (1, 2)
    netG.z_size = tmp
    ans = F.concat(output, axis=0)  # [B,2]
    ans1 = ans + 511  # bottom-right
    return F.concat([ans, ans1], axis=1)  # (B,4)

def eval_distance(pred, gt):  # (4, )
    assert len(pred.shape) == 1
    return np.linalg.norm(pred[0:2]-gt[0:2], ord=2)

@MODELS.register_module()
class BasicMatchingV2(BaseModel):
    allowed_metrics = {'dis': eval_distance}

    def __init__(self, generator, train_cfg=None, eval_cfg=None, pretrained=None):
        super(BasicMatchingV2, self).__init__()

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
        epoch = kwargs.get('epoch', 0)
        # print("now epoch: {}".format(epoch))
        optical = batchdata[0]  # [B ,1 , H, W]
        sar = batchdata[1]
        
        optical = ensemble_forward(optical, Type=epoch)
        sar = ensemble_forward(sar, Type=epoch)

        class_id = batchdata[-2]
        file_id = batchdata[-1]
        
        pre_bbox = test_generator_batch(optical, sar, netG=self.generator)  # [B, 4]

        pre_bbox = mge.tensor(bbox_ensemble_back(pre_bbox, Type=epoch))

        save_image_flag = kwargs.get('save_image')
        if save_image_flag:
            save_path = kwargs.get('save_path', None)
            start_id = kwargs.get('sample_id', None)
            if save_path is None or start_id is None:
                raise RuntimeError("if save image in test_step, please set 'save_path' and 'sample_id' parameters")
            
            with open(os.path.join(save_path, "result_epoch_{}.txt".format(epoch)), 'a+') as f:
                for idx in range(pre_bbox.shape[0]):
                    # imwrite(tensor2img(optical[idx], min_max=(-0.64, 1.36)), file_path=os.path.join(save_path, "idx_{}.png".format(start_id + idx)))
                    # 向txt中加入一行
                    suffix = ".tif"
                    write_str = ""
                    write_str += str(class_id[idx])
                    write_str += " "
                    write_str += str(class_id[idx])
                    write_str += "_"
                    write_str += str(file_id[idx]) + suffix
                    write_str += " "
                    write_str += str(class_id[idx])
                    write_str += "_sar_"
                    write_str += str(file_id[idx]) + suffix
                    write_str += " "
                    write_str += str(pre_bbox[idx][1].item())
                    write_str += " "
                    write_str += str(pre_bbox[idx][0].item())
                    write_str += "\n"
                    f.write(write_str)

        return [pre_bbox, ]

    def cal_for_eval(self, gathered_outputs, gathered_batchdata):
        """

        :param gathered_outputs: list of variable, [pre_bbox, ]
        :param gathered_batchdata: list of numpy, [optical, sar, bbox_gt, class_id, file_id]
        :return: eval result
        """
        pre_bbox = gathered_outputs[0]
        bbox_gt = gathered_batchdata[2]
        class_id = gathered_batchdata[-2]
        file_id = gathered_batchdata[-1]
        assert list(bbox_gt.shape) == list(pre_bbox.shape), "{} != {}".format(list(bbox_gt.shape), list(pre_bbox.shape))

        res = []
        sample_nums = pre_bbox.shape[0]
        for i in range(sample_nums):
            eval_result = dict()
            for metric in self.eval_cfg.metrics:
                eval_result[metric] = self.allowed_metrics[metric](pre_bbox[i].numpy(), bbox_gt[i])
            eval_result['class_id'] = class_id[i]
            eval_result['file_id'] = file_id[i]
            res.append(eval_result)
        return res

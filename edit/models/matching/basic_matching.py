import os
import time
import numpy as np
import random
import cv2
from megengine.jit import trace, SublinearMemoryConfig
import megengine.distributed as dist
import megengine as mge
import megengine.functional as F
from megengine.autodiff import GradManager
from edit.utils import imwrite, tensor2img, bgr2ycbcr, imrescale, ensemble_forward, bbox_ensemble_back
from ..base import BaseModel
from ..builder import build_backbone
from ..registry import MODELS

def add_H_W_Padding(x, margin=1):
    shape = x.shape
    padding_shape = list(shape)[:-2] + [ shape[-2] + 2*margin, shape[-1] + 2*margin ]
    res = mge.ones(padding_shape, dtype=x.dtype) * 10000
    res = res.set_subtensor(x)[:, :, margin:margin + shape[-2],  margin: margin + shape[-1]]
    return res

def get_box(xy_ctr, offsets):
    """
        xy_ctr: [1,2,37,37]
        offsets: [B,2,37,37]
    """
    xy0 = (xy_ctr - offsets)  # top-left
    xy1 = xy0 + 511  # bottom-right
    bboxes_pred = F.concat([xy0, xy1], axis=1)  # (B,4,H,W)
    return bboxes_pred

def train_generator_batch(optical, sar, label, cls_id, file_id, *, gm, netG):
    netG.train()
    with gm:
        cls_score, offsets = netG(sar, optical)
        loss, loss_cls, loss_reg, cls_labels = netG.loss(cls_score, offsets, label)
        gm.backward(loss)
        
        if dist.is_distributed():
            # do all reduce mean
            pass

    # performance in the training data
    B, _, H, W = cls_score.shape
    cls_score = cls_score.reshape(B, -1)
    times = 1
    pred_box = get_box(netG.fm_ctr, offsets)  # (B,4,H,W)
    # margin = 1
    # offsets = add_H_W_Padding(offsets, margin=margin) - (netG.z_size-1)/2
    # cls_score = add_H_W_Padding(cls_score, margin=margin)
    # cls_labels = add_H_W_Padding(cls_labels, margin=margin)

    res = []
    for t in range(times):
        output = []
        max_id = F.argmax(cls_score, axis = 1)  # (B, )
        for i in range(B):
            H_id = max_id[i] // H
            W_id = max_id[i] % H
            output.append(F.expand_dims(pred_box[i, :, H_id, W_id], axis=0))  # (1,4)
            # x = mge.tensor(-100000.0)
            # cls_score = cls_score.set_subtensor(x)[i, max_id[i]]
            # min_id = F.argmin((offsets[i, 0, H_id:H_id+(margin*2+1), W_id:W_id+(margin*2+1)]**2 + offsets[i, 1, H_id:H_id+(margin*2+1), W_id:W_id+(margin*2+1)]**2).reshape((margin*2+1)**2), axis = 0)
            # H_min_id = min_id // (margin*2+1)
            # W_min_id = min_id % (margin*2+1)
            # H_min_id = margin
            # W_min_id = margin   
            # output.append(F.expand_dims(pred_box[i, :, H_id+H_min_id-margin, W_id+W_min_id-margin], axis=0)) # (1, 4)
        output = F.concat(output, axis=0)  # (B, 4)
        res.append(output)
    output = sum(res) / len(res)

    dis = F.norm(F.floor(output[:, 0:2]+0.5) - label[:, 0:2], ord=2, axis = 1)  # (B, )
    # if F.max(dis).item() > 20:
    #     Id = F.argmax(dis)
    #     print(cls_id[Id], file_id[Id])
    #     print(dis)
    return [loss_cls*1000, loss_reg, dis.mean()]


@trace(symbolic=True)
def test_generator_batch(optical, sar, *, netG):
    netG.eval()
    tmp = netG.z_size
    netG.z_size = netG.test_z_size
    cls_score, offsets = netG(sar, optical)  # [B,1,19,19]  [B,2,19,19]  [B,1,19,19]
    B, _, H, W = cls_score.shape
    cls_score = cls_score.reshape((B, -1))
    times = 1
    pred_box = get_box(netG.test_fm_ctr, offsets)  # (B,4,H,W)

    res = []
    for t in range(times):
        output = []
        max_id = F.argmax(cls_score, axis = 1)  # (B, )
        for i in range(B):
            H_id = max_id[i] // H
            W_id = max_id[i] % H
            output.append(F.expand_dims(pred_box[i, :, H_id, W_id], axis=0))  # (1,4)
            # x = mge.tensor(-100000.0)
            # cls_score = cls_score.set_subtensor(x)[i, max_id[i]]
            # min_id = F.argmin((offsets[i, 0, H_id:H_id+(margin*2+1), W_id:W_id+(margin*2+1)]**2 + offsets[i, 1, H_id:H_id+(margin*2+1), W_id:W_id+(margin*2+1)]**2).reshape((margin*2+1)**2), axis = 0)
            # H_min_id = min_id // (margin*2+1)
            # W_min_id = min_id % (margin*2+1)
            # H_min_id = margin
            # W_min_id = margin   
            # output.append(F.expand_dims(pred_box[i, :, H_id+H_min_id-margin, W_id+W_min_id-margin], axis=0)) # (1, 4)
        output = F.concat(output, axis=0)  # (B, 4)
        res.append(output)
    output = sum(res) / len(res)
    netG.z_size = tmp
    return output  # [B,4]

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

        self.generator_gm = GradManager().attach(self.generator.parameters()) # 定义一个求导器，将指定参数与求导器绑定 

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
        optical, sar, label, cls_id, file_id = batchdata
        # 保存optical 和 sar，看下对不对
        # name = random.sample('zyxwvutsrqponmlkjihgfedcba', 3)
        # name = "".join(name) + "_" + str(label[0][0]) + "_" + str(label[0][1]) + "_" + str(label[0][2]) + "_" + str(label[0][3])
        # imwrite(cv2.rectangle(tensor2img(optical[0, ...], min_max=(-0.64, 1.36)), (label[0][1], label[0][0]), (label[0][3], label[0][2]), (0,0,255), 2), file_path="./workdirs/" + name + "_opt.png") 
        # imwrite(tensor2img(sar[0, ...], min_max=(-0.64, 1.36)), file_path="./workdirs/" + name + "_sar.png")
        optical_tensor = mge.tensor(optical, dtype="float32")
        sar_tensor = mge.tensor(sar, dtype="float32")
        label_tensor = mge.tensor(label, dtype="float32")   
        loss = train_generator_batch(optical_tensor, sar_tensor, label_tensor, cls_id, file_id, gm=self.generator_gm, netG=self.generator)
        self.optimizers['generator'].step()
        self.optimizers['generator'].clear_grad()
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
        
        optical_tensor = mge.tensor(optical, dtype="float32")
        sar_tensor = mge.tensor(sar, dtype="float32")
        pre_bbox = test_generator_batch(optical_tensor, sar_tensor, netG=self.generator)  # [B, 4]

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
        pre_bbox = F.floor(gathered_outputs[0]+0.5)
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

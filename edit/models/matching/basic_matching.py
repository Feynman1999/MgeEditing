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

epoch_dict = {}

def adjust_learning_rate(optimizer, epoch):
    if epoch % 30 == 0  and epoch_dict.get(epoch, None) is None:
        epoch_dict[epoch] = True
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * 0.9
        print("adjust lr! , now lr: {}".format(param_group["lr"]))

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
            pass

    # performance in the training data
    B, _, H, W = cls_score.shape
    cls_score = cls_score.reshape(B, -1)

    pred_box = get_box(netG.fm_ctr, offsets)  # (B,4,H,W)
    output = []
    max_id = F.argmax(cls_score, axis = 1)  # (B, )
    for i in range(B):
        H_id = max_id[i] // H
        W_id = max_id[i] % H
        output.append(F.expand_dims(pred_box[i, :, H_id, W_id], axis=0))  # (1,4)
    output = F.concat(output, axis=0)  # (B, 4)

    dis = F.norm(F.floor(output[:, 0:2] + 0.5) - label[:, 0:2], ord = 2, axis = 1)  # (B, )
    return [loss_cls*1000, loss_reg, dis.mean()]

def test_generator_batch(optical, sar, *, netG):
    netG.eval()
    tmp = netG.z_size
    netG.z_size = netG.test_z_size
    cls_score, offsets = netG(sar, optical)  # [B,1,19,19]  [B,2,19,19]  [B,1,19,19]
    B, _, H, W = cls_score.shape
    cls_score = cls_score.reshape((B, -1))

    pred_box = get_box(netG.test_fm_ctr, offsets)  # (B,4,H,W)

    output = []
    max_id = F.argmax(cls_score, axis = 1)  # (B, )
    for i in range(B):
        H_id = max_id[i] // H
        W_id = max_id[i] % H
        output.append(F.expand_dims(pred_box[i, :, H_id, W_id], axis=0))  # (1,4)
    output = F.concat(output, axis=0)  # (B, 4)

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
        # for item,p in list(self.generator.named_parameters()):
        #     print(item, p.shape)
        self.generator_gm = GradManager().attach(self.generator.parameters()) # 定义一个求导器，将指定参数与求导器绑定 

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
        optical = batchdata['opt']
        sar = batchdata['sar']
        label = batchdata['bbox']
        cls_id = batchdata['class_id']
        file_id = batchdata['file_id']

        # 保存optical 和 sar，看下对不对
        # name = random.sample('zyxwvutsrqponmlkjihgfedcba', 3)
        # name = "".join(name) + "_" + str(label[0][0]) + "_" + str(label[0][1]) + "_" + str(label[0][2]) + "_" + str(label[0][3])
        # imwrite(cv2.rectangle(tensor2img(optical[0, ...], min_max=(-0.64, 1.36)), (label[0][1], label[0][0]), (label[0][3], label[0][2]), (0,0,255), 2), file_path="./workdirs/" + name + "_opt.png") 
        # imwrite(tensor2img(sar[0, ...], min_max=(-0.64, 1.36)), file_path="./workdirs/" + name + "_sar.png")
        optical_tensor = mge.tensor(optical, dtype="float32")
        sar_tensor = mge.tensor(sar, dtype="float32")
        label_tensor = mge.tensor(label, dtype="float32")   
        loss = train_generator_batch(optical_tensor, sar_tensor, label_tensor, cls_id, file_id, gm=self.generator_gm, netG=self.generator)
        adjust_learning_rate(self.optimizers['generator'], now_epoch)
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
        # epoch = kwargs.get('epoch', 0)
        # print("now epoch: {}".format(epoch))
        optical = batchdata['opt']  # [B ,1 , H, W]
        sar = batchdata['sar']
        class_id = batchdata["class_id"]
        file_id = batchdata["file_id"]

        ensemble_flag = kwargs.get('ensemble', False)
        epochs = [0]
        res = []
        if ensemble_flag:
            epochs = list(range(0, 8))
        for epoch in epochs:
            optical_now = ensemble_forward(optical, Type=epoch)
            sar_now = ensemble_forward(sar, Type=epoch)
            optical_tensor = mge.tensor(optical_now, dtype="float32")
            sar_tensor = mge.tensor(sar_now, dtype="float32")
            pre_bbox = test_generator_batch(optical_tensor, sar_tensor, netG=self.generator)  # [B, 4]
            pre_bbox = mge.tensor(bbox_ensemble_back(pre_bbox, Type=epoch))
            res.append(pre_bbox)
        res = F.stack(res, axis=2) # [B,4,1] or [B, 4, 8]
        pre_bbox = F.mean(res, axis=2, keepdims=False)  # [B, 4]
        
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
        bbox_gt = gathered_batchdata["bbox"]
        class_id = gathered_batchdata["class_id"]
        file_id = gathered_batchdata["file_id"]
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

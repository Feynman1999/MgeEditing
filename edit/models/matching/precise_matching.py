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
    if epoch % 60 == 0  and epoch_dict.get(epoch, None) is None:
        epoch_dict[epoch] = True
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * 0.9
        print("adjust lr! , now lr: {}".format(param_group["lr"]))

def train_generator_batch(optical, sar, label, cls_id, file_id, *, gm, netG):
    netG.train()
    with gm:
        cls_score = netG(sar, optical)
        loss, cls_labels = netG.loss(cls_score, label)
        gm.backward(loss)
        if dist.is_distributed():
            loss = dist.functional.all_reduce_sum(loss) / dist.get_world_size()
        
    B, _, H, W = cls_score.shape
    cls_score = cls_score.reshape(B, -1)
    output = []
    max_id = F.argmax(cls_score, axis = 1)  # (B, )
    for i in range(B):
        H_id = max_id[i] // W
        W_id = max_id[i] % W
        output.append(netG.fm_ctr[0, :, H_id, W_id])  # (2, )
    output = F.stack(output, axis=0)  # (B, 2)
    dis = F.norm(F.floor(output[:, 0:2]+0.5) - label[:, 0:2], ord=2, axis = 1)  # (B, )
    
    if dist.is_distributed():
        dis = dist.functional.all_reduce_sum(dis) / dist.get_world_size()
        
    return [loss*1000, dis.mean()]


def test_generator_batch(optical, sar, *, netG):
    netG.eval()
    cls_score = netG(sar, optical)

    B, _, H, W = cls_score.shape
    cls_score = cls_score.reshape(B, -1)
    output = []
    max_id = F.argmax(cls_score, axis = 1)  # (B, )
    for i in range(B):
        H_id = max_id[i] // W
        W_id = max_id[i] % W
        output.append(netG.test_fm_ctr[0, :, H_id, W_id])
    output = F.stack(output, axis=0)  # (B, 2)
    return F.concat([output, output], axis= 1) # [B,4]


def eval_distance(pred, gt):  # (4, )
    assert len(pred.shape) == 1
    return np.linalg.norm(pred[0:2]-gt[0:2], ord=2)


@MODELS.register_module()
class PreciseMatching(BaseModel):
    allowed_metrics = {'dis': eval_distance}

    def __init__(self, generator, train_cfg=None, eval_cfg=None, pretrained=None):
        super(PreciseMatching, self).__init__()

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

        # name = random.sample('zyxwvutsrqponmlkjihgfedcba', 3)
        # name = "".join(name) + "_" + str(label[0][0]) + "_" + str(label[0][1]) + "_" + str(label[0][2]) + "_" + str(label[0][3])
        # imwrite(tensor2img(optical[0, ...], min_max=(-0.64, 1.36)), file_path="./workdirs/" + name + "_opt.png") 
        # imwrite(tensor2img(sar[0, ...], min_max=(-0.64, 1.36)), file_path="./workdirs/" + name + "_sar.png")
        
        optical_tensor = mge.tensor(optical, dtype="float32")
        sar_tensor = mge.tensor(sar, dtype="float32")
        label_tensor = mge.tensor(label, dtype="float32")   
        # print(optical_tensor[0,0:3,:,0:10,0:10])
        # print(label_tensor[0,0:3])
        # if len(optical_tensor.shape) == 5:
        #     B,X,C,H,W = optical_tensor.shape
        #     _,_,_,h,w = sar_tensor.shape
        #     optical_tensor = optical_tensor.reshape(B*X, C, H, W)
        #     sar_tensor = sar_tensor.reshape(B*X, C, h, w)
        #     label_tensor = label_tensor.reshape(B*X, 4)
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
        epoch = kwargs.get('epoch', 0)
        # print("now epoch: {}".format(epoch))
        optical = batchdata['opt']  # [B ,1 , H, W]
        sar = batchdata['sar']
        
        optical = ensemble_forward(optical, Type=epoch)
        sar = ensemble_forward(sar, Type=epoch)

        class_id = batchdata["class_id"]
        file_id = batchdata["file_id"]
        
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

import os
import time
import numpy as np
import random
import cv2
import megengine.distributed as dist
import megengine as mge
import megengine.functional as F
from edit.utils import imwrite, tensor2img, bgr2ycbcr, imrescale, ensemble_forward, bbox_ensemble_back
from ..base import BaseModel
from ..builder import build_backbone
from ..registry import MODELS

def train_generator_batch(optical, sar, label, cls_id, file_id, *, gm, netG):
    raise NotImplementedError(" do not support for train for TwoStageMatching")

def check_valid(x, more = 0):
    if x>=(4+more) and (x+511)<=(795 - more):
        return True
    return False

def get_location_by_sobel(sar):
    """
        sar: [1, 512, 512]
        sobel: []
        
    """
    sar = F.expand_dims(sar, axis=0)  # [1, 1, 512, 512]
    sobel_weight = mge.tensor(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32'))  # [3, 3]
    sobel_weight = F.expand_dims(sobel_weight, axis=[0,1]) # [1, 1, 3, 3]
    grad = F.conv2d(sar, sobel_weight, stride=1, padding=0) # [1, 1, 510, 510]
    grad = F.abs(grad)
    average_weight = mge.tensor(np.ones((1, 1, 256, 256), dtype='float32'))
    grad = F.conv2d(grad, average_weight, stride=1, padding=0)  # [1, 1, 255, 255]
    # 取最大下标
    max_id = F.argmax(grad.flatten()) # shape: ()
    return (int(max_id // 255), int(max_id % 255))

def test_generator_batch(optical, sar, *, G1, G2):
    # B,_,_,_ = optical.shape
    # return mge.tensor([0.,0.,0.,0.0]*B).reshape(B,-1)
    G1.eval()
    G2.eval()
    tmp = G1.z_size
    G1.z_size = G1.test_z_size # 512
    # stage1
    cls_score, offsets = G1(sar, optical)  # [B,1,19,19]  [B,2,19,19]  [B,1,19,19]
    B1, _, H1, W1 = cls_score.shape
    cls_score = cls_score.reshape((B1, -1))
    pred_box = G1.test_fm_ctr - offsets  # (B,2,H,W)
    output = []
    flag = []
    stage2_sar = []
    stage2_optical = []
    max_id = F.argmax(cls_score, axis = 1)  # (B1, )
    for i in range(B1):
        H_id = max_id[i] // H1
        W_id = max_id[i] % H1
        # pred_box[i, :, H_id, W_id]是预测的左上角坐标， 进行四舍五入
        top_left = F.floor(pred_box[i, :, H_id, W_id]+0.5)
        # top_left = pred_box[i, :, H_id, W_id]
        # 根据top_left，截取512周围的520（如果足够的话）
        x,y = int(top_left[0]), int(top_left[1])
        if check_valid(x) and check_valid(y):
            # method1, use 520 512 but too slow
            stage2_optical.append(optical[i, :, x-4 : x+516, y-4 : y+516])
            stage2_sar.append(sar[i, ...])

            # method2, use up left 260 256, fast but bad result  (0.8 -> 0.93)
            # stage2_optical.append(optical[i, :, x-2 : x+258, y-2 : y+258])
            # stage2_sar.append(sar[i, :, 0:256, 0:256])

            # method3, find grad change most 256 in 512 sar
            # sar_x, sar_y = get_location_by_sobel(sar[i, ...])  # [0, 254]
            # sar_256 = sar[i, :, (1+sar_x):(1+sar_x+256), (1+sar_y):(1+sar_y+256)]
            # # write sar and sar_256
            # imwrite(tensor2img(sar_256, min_max=(-0.64, 1.36)), file_path="./workdirs/{}_{}.png".format(sar_x, sar_y))
            # imwrite(tensor2img(sar[i,:,:,:], min_max=(-0.64, 1.36)), file_path="./workdirs/{}_{}_large.png".format(sar_x, sar_y))
            # optical_260 = optical[i, :, x+1+sar_x-2 : x+1+sar_x + 256 + 2, y+1+sar_y-2 : y+1+sar_y + 256 + 2]
            # stage2_optical.append(optical_260)
            # stage2_sar.append(sar_256)

            flag.append(1)
        else:
            flag.append(0)
        output.append(top_left)

    # for stage2, fix output
    if 1 in flag:
        sar = F.stack(stage2_sar, axis=0)
        optical = F.stack(stage2_optical, axis=0)
        cls_score = G2(sar, optical)
        B2, _, H2, W2 = cls_score.shape
        cls_score = cls_score.reshape((B2, -1))
        max_id = F.argmax(cls_score, axis=1) # (B2, )
        idx = 0
        for i in range(B1):
            if flag[i] == 1: # do stage2, need fix output
                H_id = max_id[idx] // H2
                W_id = max_id[idx] % H2
                print("before: {}  ".format(output[i]), end="")
                output[i] = output[i] + mge.tensor([H_id, W_id]) - mge.tensor([H2//2, H2//2])
                print("after: {}".format(output[i]))
                idx += 1
            else:
                print("do not fix")
        assert idx == B2
    else:
        print("do not have flag 1 in this batch!")
    output = F.stack(output, axis=0)  # (B, 2)
    G1.z_size = tmp
    return F.concat([output, output+511], axis= 1)  # [B,4]

def eval_distance(pred, gt):  # (2, )
    assert len(pred.shape) == 1
    return np.linalg.norm(pred[0:2]-gt[0:2], ord=2)

@MODELS.register_module()
class TwoStageMatching(BaseModel):
    """
        only for test
    """
    allowed_metrics = {'dis': eval_distance}

    def __init__(self, generator1, generator2, train_cfg=None, eval_cfg=None, pretrained=None):
        super(TwoStageMatching, self).__init__()

        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg

        # generator
        self.generator1 = build_backbone(generator1)
        self.generator2 = build_backbone(generator2)
        
        # load pretrained
        self.init_weights(pretrained)

        # self.generator_gm = GradManager().attach(self.generator.parameters()) # 定义一个求导器，将指定参数与求导器绑定

    def init_weights(self, pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        pass

    def train_step(self, batchdata):
        """train step.

        Args:
            batchdata: list for train_batch, numpy.ndarray, length up to Collect class.
        Returns:
            list: loss
        """
        print("pass for train step!")
        pass

    def test_step(self, batchdata, **kwargs):
        """test step.

        Args:
            batchdata: list for train_batch, numpy.ndarray or variable, length up to Collect class.

        Returns:
            list: outputs (already gathered from all threads)
        """
        optical = batchdata[0]  # [B ,1 , H, W]
        sar = batchdata[1]
        class_id = batchdata[-2]
        file_id = batchdata[-1]

        ensemble_flag = kwargs.get('ensemble_flag', False)
        epochs = [0]
        res = [] # item: [B,4]
        if ensemble_flag:
            epochs = list(range(0,1))
        for epoch in epochs:
            optical_now = ensemble_forward(optical, Type=epoch)
            sar_now = ensemble_forward(sar, Type=epoch)
            optical_tensor = mge.tensor(optical_now, dtype="float32")
            sar_tensor = mge.tensor(sar_now, dtype="float32")
            pre_bbox = test_generator_batch(optical_tensor, sar_tensor, G1=self.generator1, G2=self.generator2)  # [B, 4]
            pre_bbox = mge.tensor(bbox_ensemble_back(pre_bbox, Type=epoch))
            res.append(pre_bbox)
        res = F.stack(res, axis=2) # [B,4,1] or [B, 4, 8]
        #print(res[0])
        pre_bbox = F.mean(res, axis=2, keepdims=False)  # [B, 4]

        save_image_flag = kwargs.get('save_image')
        if save_image_flag:
            save_path = kwargs.get('save_path', None)
            if save_path is None:
                raise RuntimeError("if want save image(or result) in test_step, please set 'save_path' parameters")
            # todo: 每一次都打开，会不会影响速度？
            with open(os.path.join(save_path, "result_epoch_{}_rank_{}.txt".format(epoch, self.local_rank)), 'a+') as f:
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

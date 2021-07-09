import os
import time
from megengine.functional.tensor import concat
import numpy as np
import cv2
import megengine.distributed as dist
import megengine as mge
import megengine.functional as F
from megengine.autodiff import GradManager
from edit.utils import imwrite, tensor2img, bgr2ycbcr, img_multi_padding, img_de_multi_padding
from edit.utils import img_multi_padding, img_de_multi_padding
from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS
from tqdm import tqdm
from collections import defaultdict
from .centertrack import train_batch, test_batch
from tensorboardX import SummaryWriter
import random
import pandas as pd

epoch_dict = {}

def adjust_learning_rate(optimizer, epoch):
    if epoch>=50 and epoch % 15 == 0  and epoch_dict.get(epoch, None) is None:
        epoch_dict[epoch] = True
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * 0.3
        print("adjust lr! , now lr: {}".format(param_group["lr"]))

@MODELS.register_module()
class ONLINE_MOT(BaseModel):
    allowed_metrics = {}

    def __init__(self, generator, loss_weight,
                 train_cfg=None, eval_cfg=None, pretrained=None, **kwargs):
        super(ONLINE_MOT, self).__init__()
        self.loss_weight = loss_weight
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        # generator
        self.generator = build_backbone(generator)

        # load pretrained
        self.init_weights(pretrained)

        self.min_max = (-0.5, 0.5)
        print("min max: {}".format(self.min_max))

        self.train_generator_batch = train_batch
        self.test_generator_batch = test_batch

        # get workdir 
        workdir = kwargs.get("workdir", "./workdir")
        self.summary = {}
        self.gen_writer = None
        if self.local_rank == 0:
            self.gen_writer = SummaryWriter(os.path.join(workdir, 'tensorboard_gen'))
            self.gen_writer_gap = 100

    def init_weights(self, pretrained=None):
        pass
    
    def viz_frame_and_bbox(self, img, bboxes, now_iter):
        print(now_iter)
        img = tensor2img(img[0], min_max=self.min_max)
        img = img.copy()
        for i in range(len(bboxes[0])):
            img = cv2.rectangle(img, bboxes[0][i][0:2].astype(np.int32), bboxes[0][i][2:4].astype(np.int32), (0,255,255), 1)
            img = cv2.putText(img, str(i), bboxes[0][i][0:2].astype(np.int32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
       
        imwrite(img, file_path = "./workdirs/rrr/{}_img.png".format(now_iter))

    def viz_frames_and_bbox(self, img1, img2, gt_bboxes1, gt_bboxes2, gt_labels1, gt_labels2, gt_bboxes1_num, gt_bboxes2_num, now_iter):
        def num_to_color(num):
            z = num % 256
            num = num // 256
            y = num % 256
            num = num //256
            return (num, y, z)

        def get_id_to_color_dict(nums = 50):
            # 随机生成nums种颜色, (x,y,z)  [0,255]
            assert nums <= 100
            res = {}
            random.seed(23333)
            res2 = random.sample(range(0, 256**3), nums)
            for id, item in enumerate(res2):
                res[id+1] = num_to_color(item)
            return res

        color_dict = get_id_to_color_dict()

        # print(img1.shape)
        # print(img2.shape)
        # print(gt_bboxes1.dtype)
        # print(gt_bboxes2.dtype)
        # print(gt_bboxes1_num.dtype)
        # print(gt_bboxes2_num.dtype)
        img1 = tensor2img(img1[0], min_max=self.min_max)
        img2 = tensor2img(img2[0], min_max=self.min_max)
        # 分别给img1和img2打上bbox
        img1 = img1.copy()
        img2 = img2.copy()
        print(gt_bboxes1[0])
        print(gt_bboxes1_num[0])
        for i in range(gt_bboxes1_num[0]):
            assert gt_labels1[0][i][0] == 0
            id = gt_labels1[0][i][1]
            img1 = cv2.rectangle(img1, gt_bboxes1[0][i][0:2].astype(np.int32), gt_bboxes1[0][i][2:4].astype(np.int32), color_dict[id], 1)
        for i in range(gt_bboxes2_num[0]):
            assert gt_labels2[0][i][0] == 0
            id = gt_labels2[0][i][1]
            img2 = cv2.rectangle(img2, gt_bboxes2[0][i][0:2].astype(np.int32), gt_bboxes2[0][i][2:4].astype(np.int32), color_dict[id], 1)
        imwrite(img1, file_path = "./workdirs/{}_img1.png".format(now_iter))
        imwrite(img2, file_path = "./workdirs/{}_img2.png".format(now_iter))

    def add_summary(self, writer, name, val, iter):
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if writer is not None and iter % self.gen_writer_gap == 0:
            writer.add_scalar(name, self.summary[name]/self.gen_writer_gap, iter)
            self.summary[name] = 0

    def split_to_list(self, batchdata, num):
        res = []
        b,x,z = batchdata.shape
        assert z in (4, 2)
        for i in range(b):
            pre_num = num[i]
            res.append(batchdata[i, 0:pre_num, :])
        return res

    def train_step(self, batchdata, now_epoch, now_iter):
        img1 = batchdata['img1']
        img2 = batchdata['img2']
        gt_bboxes1 = batchdata['gt_bboxes1']
        gt_bboxes2 = batchdata['gt_bboxes2']
        gt_labels1 = batchdata['gt_labels1']
        gt_labels2 = batchdata['gt_labels2']
        gt_bboxes1_num = batchdata['gt_bboxes1_num']
        gt_bboxes2_num = batchdata['gt_bboxes2_num']
        # 根据num把bbox和labels拆成列表，每一个取前num个
        new_gt_bboxes1 = self.split_to_list(gt_bboxes1, gt_bboxes1_num)
        new_gt_labels1 = self.split_to_list(gt_labels1, gt_bboxes1_num)
        new_gt_bboxes2 = self.split_to_list(gt_bboxes2, gt_bboxes2_num)
        new_gt_labels2 = self.split_to_list(gt_labels2, gt_bboxes2_num)
        img1 = mge.tensor(img1, dtype="float32")
        img2 = mge.tensor(img2, dtype="float32")

        # self.viz_frames_and_bbox(img1, img2, gt_bboxes1, gt_bboxes2, gt_labels1, gt_labels2, gt_bboxes1_num, gt_bboxes2_num, now_iter)

        loss = self.train_generator_batch(img1, img2, new_gt_bboxes1, new_gt_bboxes2, new_gt_labels1, new_gt_labels2,
                                          gm_G=self.gms['generator'], netG=self.generator, 
                                          optim_G = self.optimizers['generator'],
                                          loss_weight = self.loss_weight, now_iter = now_iter)
        adjust_learning_rate(self.optimizers['generator'], now_epoch)
        self.add_summary(self.gen_writer, 'loss/heatmap_loss', loss[0].item(), now_iter)
        self.add_summary(self.gen_writer, 'loss/hw_loss', loss[1].item(), now_iter)
        self.add_summary(self.gen_writer, 'loss/motion_loss', loss[2].item(), now_iter)
        return loss

    def save_track_info(self, frameNo, bbox, identities=None):
        cur_frame_track_info = []
        for i, box in enumerate(bbox):
            id = int(identities[i]) if identities is not None else 0
            x1, y1, x2, y2 = [int(i+0.5) for i in box]

            cur_frame_track_info.append({
                'frameNo': frameNo,
                'trackid': id,
                'boxesX1': x1,
                'boxesY1': y1,
                'boxesX2': x2,
                'boxesY2': y2,
                'conf': 0,
                'cat': 1,
                'iscrowd': 0,
            })
        print(f"{frameNo} is ok")
        self.all_track_info.extend(cur_frame_track_info)

    def test_step(self, batchdata, **kwargs):
        """
            possible kwargs:
                save_image
                save_path
                ensemble
        """
        img = batchdata['img']
        index = batchdata['index']
        clip = batchdata['clipname']
        scale_factor = batchdata['scale_factor']
        gap = batchdata['gap']
        total_len = batchdata['total_len'][0]
        """
            如果是第一帧则生成id，否则根据前一帧各个id的位置，为当前帧分配id
        """
        if index[0] == 0:
            self.all_track_info = []
            # 每一次如果有结果，就放到这里
            self.pre_bboxes = None # List len: batchsize [[S,4]]
            self.pre_img = mge.tensor(img, dtype="float32")
            self.pre_labels = None
        # now_labels: List len: batchsize [[S,2]]
        now_bboxes, now_labels = test_batch(img1 = self.pre_img, img2 = mge.tensor(img, dtype="float32"), 
                   pre_bboxes=self.pre_bboxes, netG = self.generator, pre_labels = self.pre_labels, gap=int(gap[0]))
        self.pre_bboxes = now_bboxes
        self.pre_img = mge.tensor(img, dtype="float32")
        self.pre_labels = now_labels
        # for item in now_labels[0]:
        #     print(item, end= "")
        # print("\n ")
        # self.viz_frame_and_bbox(img, now_bboxes, index[0])

        #　对bboxes进行scale, 到当前帧的原输入大小
        write_bbox = now_bboxes[0] # [s,4]
        write_bbox = write_bbox / scale_factor
        write_label = now_labels[0] # [s,2]
        # 将write_bbox和write_label加入all_track_info中
        self.save_track_info(frameNo=index[0]+1, bbox = write_bbox, identities=write_label[:, 1])
        
        if index[0] == total_len - 1:
            # write all_track_info to txt
            track_result_path = os.path.join(
                kwargs.get('save_path', './workdirs'), f'{clip[0]}_track_s{gap[0]}_test_no1.txt'
            )
            df = pd.DataFrame(self.all_track_info)
            df.to_csv(track_result_path, index=False, header=False)
            return True
        return False

    def cal_for_eval(self, gathered_outputs, gathered_batchdata):
        if gathered_outputs:
            raise NotImplementedError("")
        else:
            return []

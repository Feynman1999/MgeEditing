import os
import time
from megengine import tensor
import numpy as np
import megengine.distributed as dist
import megengine as mge
import megengine.functional as F
from megengine.autodiff import GradManager
from edit.core.hook.evaluation import psnr, ssim
from edit.utils import imwrite, tensor2img, bgr2ycbcr, img_multi_padding, img_de_multi_padding, ensemble_forward, ensemble_back
from edit.utils import img_multi_padding, img_de_multi_padding
from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS
from tqdm import tqdm
from collections import defaultdict
from .sttn import train_batch, test_batch
from tensorboardX import SummaryWriter

epoch_dict = {}

def adjust_learning_rate(optimizer, epoch):
    if epoch>=600 and epoch % 20 == 0  and epoch_dict.get(epoch, None) is None:
        epoch_dict[epoch] = True
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * 0.8
        print("adjust lr! , now lr: {}".format(param_group["lr"]))

@MODELS.register_module()
class STTN_synthesizer(BaseModel):
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self, generator, discriminator, pixel_loss, adv_loss, loss_weight,
                 train_cfg=None, eval_cfg=None, pretrained=None, **kwargs):
        super(STTN_synthesizer, self).__init__()
        self.loss_weight = loss_weight
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        # generator
        self.generator = build_backbone(generator)
        # discri
        self.discriminator = build_backbone(discriminator)
        # loss
        self.pixel_loss = build_loss(pixel_loss)
        self.adv_loss = build_loss(adv_loss)
        # load pretrained
        self.init_weights(pretrained)

        self.min_max = (-1, 1)
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

        self.now_deal_clip = -1 # for test or eval

    def init_weights(self, pretrained=None):
        # 对G使用pretrained
        # self.generator.init_weights(pretrained = True)
        pass
    
    def viz_frames_and_masks(self, frames, masks):
        masked_frames = (frames * (1 - masks))
        frames = masked_frames[0] # [t,c,h,w]
        masks = masks[0] # [t,c,h,w]
        for i in range(5):
            imwrite(tensor2img(frames[i], min_max=self.min_max), file_path = "./workdirs/{}.png".format(i))
            imwrite(tensor2img(masks[i]), file_path = "./workdirs/{}_mask.png".format(i))

    def add_summary(self, writer, name, val, iter):
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if writer is not None and iter % self.gen_writer_gap == 0:
            writer.add_scalar(name, self.summary[name]/self.gen_writer_gap, iter)
            self.summary[name] = 0

    def train_step(self, batchdata, now_epoch, now_iter):
        frames = batchdata['frames']
        masks = batchdata['masks']
        frames = mge.tensor(frames, dtype="float32")
        masks = mge.tensor(masks, dtype="float32")
        # self.viz_frames_and_masks(frames, masks)
        loss = self.train_generator_batch(frames, masks, 
                                          gm_G=self.gms['generator'], netG=self.generator, 
                                          gm_D=self.gms['discriminator'], netD = self.discriminator,
                                          optim_G = self.optimizers['generator'], optim_D = self.optimizers['discriminator'],
                                          pixel_loss=self.pixel_loss, adv_loss = self.adv_loss, loss_weight = self.loss_weight)
        adjust_learning_rate(self.optimizers['generator'], now_epoch)
        adjust_learning_rate(self.optimizers['discriminator'], now_epoch)
        self.add_summary(self.gen_writer, 'loss/hole_loss', loss[0].item(), now_iter)
        self.add_summary(self.gen_writer, 'loss/valid_loss', loss[1].item(), now_iter)
        return loss

    def imgs_to_video(self, save_path, clipname, imgs):
        pass

    def read_bbox(self):
        path = "/data/home/songtt/chenyuxiang/datasets/mgtv/test/test_a"
        dir_name = "video_{}".format(str(self.now_deal_clip).zfill(4))
        self.bboxes = []
        with open(os.path.join(path, dir_name, "minbbox.txt"), "r") as f:
            for line in f.readlines():
                if line.strip() == "":
                    continue
                x,y,w,h = [int(item) for item in line.strip().split(" ")]
                self.bboxes.append((x,y,w,h))
        assert len(self.bboxes) == len(self.img_list)
        print("load {} ok!".format(os.path.join(path, dir_name, "minbbox.txt")))

    def solve(self, start, len, save_path):
        """
            从index为start开始取，长度为len，另外以gap = 20从两边取一些帧
            先做个实验，不取较远帧
        """
        img = np.stack(self.img_list[start:start + len], axis=1) # [1,5,c,h,w]
        mask = np.stack(self.mask_list[start:start + len], axis=1)
        img = mge.tensor(img, dtype="float32")
        mask = mge.tensor(mask, dtype="float32")
        res = test_batch(img, mask, self.generator)
        for i in range(start, start + len):
            print("write clip: {} idx: {}".format(self.now_deal_clip, i))
            x,y,w,h = self.bboxes[i]
            imwrite(tensor2img(res[i-start], min_max = self.min_max)[y:y+h, x:x+w, :], os.path.join(save_path, f"video_{str(self.now_deal_clip).zfill(4)}",
                       f"crop_{str(i).zfill(6)}.png"))

    def test_step(self, batchdata, **kwargs):
        """
            batchdata keys:
                img   (b,3,h,w)
                mask  (b,1,h,w) need to 0 or 1
                index            (1,)  当前idx 从0开始
                max_frame_num    (1,)  一共帧数
            possible kwargs:
                save_image
                save_path
                ensemble
        """
        img = batchdata['img']
        mask = batchdata['mask']
        index = batchdata['index']
        max_frame_num = batchdata['max_frame_num'][0]
        bs, _, _, _ = img.shape
        assert bs == 1

        if index[0] == 0:
            """
                第一帧
            """
            self.now_deal_clip += 1
            self.img_list = []
            self.mask_list = []
            
        """
            对于每一帧，将当前帧放到img_list中
        """
        self.img_list.append(img)
        self.mask_list.append(mask)

        if index[0] == max_frame_num - 1:
            """
                读取所有帧的bbox, 存到self中
                最后一帧, 开始处理并保存
            """
            self.read_bbox()
            for i in range(0, max_frame_num//5):
                self.solve(i*5, 5, save_path = kwargs.get("save_path", "./workdirs"))
            if max_frame_num % 5 != 0:
                self.solve(max_frame_num//5 * 5, max_frame_num % 5, save_path = kwargs.get("save_path", "./workdirs"))
            return True
        return False

    def cal_for_eval(self, gathered_outputs, gathered_batchdata):
        if gathered_outputs:
            raise NotImplementedError("")
        else:
            return []

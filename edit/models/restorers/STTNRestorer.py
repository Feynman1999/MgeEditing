import os
import time
import numpy as np
import megengine.distributed as dist
import megengine as mge
import megengine.functional as F
from collections import defaultdict
from megengine.autodiff import GradManager
from edit.core.hook.evaluation import psnr, ssim
from edit.utils import imwrite, tensor2img, bgr2ycbcr, img_multi_padding, img_de_multi_padding, ensemble_forward, ensemble_back
from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS

def get_bilinear(image):
    B,T,C,h,w = image.shape
    image = image.reshape(-1, C,h,w)
    return F.nn.interpolate(image, scale_factor=4).reshape(B,T,C,4*h, 4*w)

def train_generator_batch(image, label, *, gm, netG, netloss):
    B,T,C,h,w = image.shape
    biup = get_bilinear(image)
    netG.train()
    with gm:
        output = netG(image)
        loss = netloss(output + biup, label)
        gm.backward(loss)
        if dist.is_distributed():
            loss = dist.functional.all_reduce_sum(loss) / dist.get_world_size()
    return loss

def test_generator_batch(image, *, netG):
    biup = get_bilinear(image)
    netG.eval()
    output = netG(image)
    return output + biup

@MODELS.register_module()
class STTNRestorer(BaseModel):
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self, generator, pixel_loss, train_cfg=None, eval_cfg=None, pretrained=None):
        super(STTNRestorer, self).__init__()

        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg

        # generator
        self.generator = build_backbone(generator)
        # loss
        self.pixel_loss = build_loss(pixel_loss)

        # load pretrained
        self.init_weights(pretrained)

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
        LR_tensor = mge.tensor(batchdata['lq'], dtype="float32")
        HR_tensor = mge.tensor(batchdata['gt'], dtype="float32")
        # B,t,c,h,w = LR_tensor.shape
        # for i in range(B):
        #     for t in range(t):
        #         print(i, t)
        #         imwrite(tensor2img(LR_tensor[i,t], min_max=(-0.5, 0.5)), file_path="./haha/LR_{}_{}.png".format(i, t))
        #         imwrite(tensor2img(HR_tensor[i,t], min_max=(-0.5, 0.5)), file_path="./haha/HR_{}_{}.png".format(i, t))
        loss = train_generator_batch(LR_tensor, HR_tensor, gm=self.gms['generator'], netG=self.generator, netloss=self.pixel_loss)
        # for key,_ in self.generator.encoder.named_parameters():
        #     print(key)
        # print(self.generator.encoder[0].weight.grad)
        self.optimizers['generator'].step()
        self.optimizers['generator'].clear_grad()
        return loss

    def get_img_id(self, key):
        shift = self.eval_cfg.get('save_shift', 0)
        assert isinstance(key, str)
        L = key.split("/")
        return int(L[-1][:-4]), str(int(L[-2]) - shift).zfill(3) # id, clip

    def test_step(self, batchdata, **kwargs):
        start_time = time.time()
        lq = batchdata['lq']
        if len(lq.shape) == 4:
            lq = np.expand_dims(lq, 1) # [B,T,C,H,W]
        gt = batchdata['gt'][0]
        lq_paths = [item[0] for item in batchdata['lq_path']]
        num_input_frames =  batchdata['num_input_frames'][0]

        # get ids
        ids = [ tuple(self.get_img_id(path))[0] for path in lq_paths]
        clip = self.get_img_id(lq_paths[0])[1]

        B,T,_,h,w = lq.shape
        assert B == 1, "only support batchsize==1 for test and eval now"

        if ids[ num_input_frames//2 ] == 0:
            print("first frame: {}".format(batchdata['LRkey'][0]))
            self.HR_frame_dict = defaultdict(list)  # 'LR_key -> list of tensor'            
            self.GT_frame_dict = {}

        self.GT_frame_dict[ids[ num_input_frames//2 ]] = gt

        lq_tensor = mge.tensor(lq, dtype="float32") # [B,T,C,H,W]
        output = test_generator_batch(lq_tensor, netG = self.generator)
        G = output[0, ...]  # [T,3,H,W]
        # get result
        for i in range(num_input_frames):
            self.HR_frame_dict[ ids[i] ].append(G[i, ...].numpy())

        if kwargs.get('save_image', False):
            print("saving images to disk ...")
            save_path = kwargs.get('save_path', None)
            if save_path is None:
                raise RuntimeError("if save image in test_step, please set 'save_path' parameters")
            imwrite(tensor2img(G[num_input_frames//2, ...], min_max=(-0.5, 0.5)), file_path=os.path.join(save_path, "{}_{}.png".format(clip, ids[num_input_frames//2])))

        print("imgs {} ok  inference time: {} s".format(ids, time.time() - start_time))
        return ids[ num_input_frames//2 ] == 99

    def cal_for_eval(self, gathered_outputs, gathered_batchdata):
        """
            :param gathered_outputs: list of tensor, [B,C,H,W]
            :param gathered_batchdata: dict, include data
            :return: eval result
        """
        if gathered_outputs:
            # 当前是最后一帧，此时计算
            crop_border = self.eval_cfg.crop_border
            assert len(self.HR_frame_dict.keys()) == 100
            res = []
            for key in sorted(self.HR_frame_dict.keys()):
                print("average {} times for {}".format(len(self.HR_frame_dict[key]), key))
                G_key = np.mean(np.stack(self.HR_frame_dict[key], axis=0), axis=0, keepdims=False)
                G_key = tensor2img(G_key, min_max=(-0.5, 0.5))
                # G_key_y = bgr2ycbcr(G_key, y_only=True)
                gt = self.GT_frame_dict[key]
                gt = tensor2img(gt, min_max=(-0.5, 0.5))
                # gt_y = bgr2ycbcr(gt, y_only=True)

                eval_result = dict()
                for metric in self.eval_cfg.metrics:
                    eval_result[metric+"_RGB"] = self.allowed_metrics[metric](G_key, gt, crop_border)
                    # eval_result[metric+"_Y"] = self.allowed_metrics[metric](G_key_y, gt_y, crop_border)
                res.append(eval_result)
            return res
        else:
            return []

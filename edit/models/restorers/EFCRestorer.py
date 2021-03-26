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
from skimage.segmentation import slic, mark_boundaries

def train_generator_batch(image, label, trans_flags, *, gm, netG, netloss):
    """
        image: list of tensor, (1,3,h,w)
        label: list of tensor, (1,3,H,W)
    """
    T = len(image)
    biup = [ F.nn.interpolate(item, scale_factor=4) for item in image]
    netG.train()
    with gm:
        output = netG(image, trans_flags)
        for idx in range(T):
            output[idx] += biup[idx]
        loss = netloss(output, label)
        gm.backward(loss)
        if dist.is_distributed():
            loss = dist.functional.all_reduce_sum(loss) / dist.get_world_size()
    # 可视化三个东西
    # LR G HR
    # global now_viz
    # for i in range(T):
    #     # imwrite(tensor2img(biup[i][0, ...], min_max=(-0.5, 0.5)), file_path="./viz/{}/LR_{}.png".format(now_viz,i))
    #     # imwrite(tensor2img(output[i][0, ...], min_max=(-0.5, 0.5)), file_path="./viz/{}/G_{}.png".format(now_viz,i))
    #     imwrite(tensor2img(F.concat([biup[i][0, ...], output[i][0, ...], label[i][0, ...]], axis=1), min_max=(-0.5, 0.5)), file_path="./viz/{}/{}.png".format(now_viz,i))
    # now_viz+=1
    return loss

def test_generator_batch(image, *, netG):
    T = len(image)
    biup = [ F.nn.interpolate(item, scale_factor=4) for item in image]
    netG.eval()
    output = netG(image, [False])
    for idx in range(T):
        output[idx] += biup[idx]
    return output # list of tensor

@MODELS.register_module()
class EFCRestorer(BaseModel):
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self, generator, pixel_loss, train_cfg=None, eval_cfg=None, pretrained=None):
        super(EFCRestorer, self).__init__()

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
        LR_tensor =  [mge.tensor(item, dtype="float32") for item in batchdata['lq']]
        HR_tensor =  [mge.tensor(item, dtype="float32") for item in batchdata['gt']]
        # for idx, item in enumerate(LR_tensor):
        #     imwrite(tensor2img(item[0]), file_path="./viz_efc/{}/LR_{}.png".format(self.xxx, idx))
        #     imwrite(tensor2img(HR_tensor[idx][0]), file_path="./viz_efc/{}/HR_{}.png".format(self.xxx, idx))
        # self.xxx += 1            
        # B,t,c,h,w = LR_tensor.shape
        # for i in range(B):
        #     for t in range(t):
        #         print(i, t)
        #         imwrite(tensor2img(LR_tensor[i,t], min_max=(-0.5, 0.5)), file_path="./haha/LR_{}_{}.png".format(i, t))
        #         imwrite(tensor2img(HR_tensor[i,t], min_max=(-0.5, 0.5)), file_path="./haha/HR_{}_{}.png".format(i, t))
        loss = train_generator_batch(LR_tensor, HR_tensor, batchdata['transpose'], gm=self.gms['generator'], netG=self.generator, netloss=self.pixel_loss)
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

    def check_valid(self, x):
        if x[0]<0 or x[0] >= 180:
            return False
        if x[1]<0 or x[1] >= 320:
            return False
        return True

    def test_step_block_by_block(self, inp, ans, clip):
        """
            inp: (100, 3, now_h, now_w) ndarray
            ans: (1, 100, 3, 4*now_h, 4*now_w) ndarray, need to be filled
        """
        # 20帧一组，分4*4块处理，结果放到ans中
        T, _, now_h, now_w = inp.shape
        method = self.eval_cfg.get("method", "flow_crop")
        assert method in ("normal_crop", "flow_crop") 
        if method == "normal_crop":
            gap = 1
            block_h = now_h // gap
            block_w = now_w // gap
            for seg in range(0,T,20):# seg: seg+20
                for i in range(gap):
                    for j in range(gap):
                        print("seg: {}-{},  i:{},  j:{}".format(seg, seg+20-1, i, j))
                        inp_list = []
                        for t in range(seg, seg+20):
                            inp_list.append(mge.tensor(inp[t:t+1, :, i*block_h:(i*block_h+block_h), j*block_w:(j*block_w+block_w)], dtype="float32"))
                        oup_list = test_generator_batch(inp_list, netG = self.generator)
                        # 可视化
                        # if i==0 and j==0:
                        #     imwrite(tensor2img(oup_list[0][0, ...], min_max=(-0.5, 0.5)), file_path="./xxx_seg_{}.png".format(seg))
                        # set ans
                        for t in range(20):
                            ans[0:1, seg + t, :, i*4*block_h:(i*4*block_h+4*block_h), j*4*block_w:(j*4*block_w+4*block_w)] = oup_list[t].numpy()
        else: # flow_crop
            tong = np.zeros_like(ans) # 计数矩阵,统计每个像素点被算了多少次，最后算平均
            tong = np.mean(tong, axis=2, keepdims=True) # [1, 100, 1, 4*h, 4*w]
            n_segments = 160 # super params of slic
            compactness = 15 # super params of slic
            name_padding_len = 8
            threthld = 8*9
            blocksizes = [9, 8]
            ref_len = 19
            flow_dir = "/work_base/datasets/REDS/train/train_sharp_bicubic/X4_RAFT_sintel"
            for t in range(T): # 确保这一帧已经被填满后，进入下一帧
                # 对当前帧进行slic算法
                segments_lq_first_frame = slic(tensor2img(inp[t, ...], min_max=(-0.5, 0.5)), n_segments=n_segments, compactness=compactness, start_label=0) # [180, 320]
                max_class_id = np.max(segments_lq_first_frame)
                # 对于ans中计数为0的部分，选择一块，进行推理, 注意坐标除4
                for i in range(now_h):
                    for j in range(now_w):
                        # check tong of now frame
                        if tong[0, t, 0, 4*i, 4*j] < 0.5: # 不用==0 防止精度出问题
                            print("now deal t: {},  i: {},  j: {}".format(t, i, j))
                            # 选择当前帧对应的class_id
                            select_class_id = segments_lq_first_frame[i, j]
                            # 往后一共选19帧或者因过小提前结束
                            lq_masks = []
                            mask_lq_first_frame = np.argwhere(select_class_id == segments_lq_first_frame)  # e.g. (672, 2) int64
                            lq_masks.append(mask_lq_first_frame)
                            first_frame_idx = t
                            for idx in range(first_frame_idx, min(T-1, first_frame_idx + ref_len - 1)):
                                # according    idx -> idx+1      flow   solve   idx+1  mask
                                flowpath = os.path.join(flow_dir, clip, "{}_{}.npy".format(str(idx).zfill(name_padding_len), str(idx+1).zfill(name_padding_len)))
                                flow = np.load(flowpath)
                                L = []
                                for h,w in lq_masks[-1]:
                                    res = [int(flow[h,w,1]+0.5) + h, int(flow[h,w,0]+0.5) + w]
                                    if self.check_valid(res):
                                        L.append(res)
                                if len(L) < threthld:
                                    break
                                new_mask = np.array(L)
                                lq_masks.append(new_mask)
                            # crop for lq
                            lq_crops = []
                            record_tl = []
                            record_length_h = []
                            record_length_w = []
                            for idx in range(0, len(lq_masks)):
                                tl = np.min(lq_masks[idx], axis=0) # top-left
                                br = np.max(lq_masks[idx], axis=0) # bottom-right
                                # make tl and br    are integral multiple of the block size
                                tl = (np.floor(tl / blocksizes) * blocksizes ).astype(np.int64)
                                br = (np.ceil((br+1) / blocksizes) * blocksizes ).astype(np.int64) - 1
                                length_h = br[0] - tl[0] + 1
                                length_w = br[1] - tl[1] + 1
                                lq_crops.append(mge.tensor(inp[(idx+t):(idx+t+1), :, tl[0]:tl[0] + length_h, tl[1]:tl[1] + length_w], dtype="float32"))
                                # 更新tong
                                tong[0, idx + t, 0, 4*tl[0]:4*(tl[0] + length_h), 4*tl[1]: 4*(tl[1] + length_w)] += 1.0
                                # record tl
                                record_tl.append(tl)
                                record_length_h.append(length_h)
                                record_length_w.append(length_w)
                            # 处理lq_crops，并将结果加到ans上
                            print(lq_crops[0].shape)
                            oup_list = test_generator_batch(lq_crops, netG = self.generator)
                            # 将结果加到ans上
                            for idx in range(0, len(oup_list)):
                                tl = record_tl[idx]
                                length_h = record_length_h[idx]
                                length_w = record_length_w[idx]
                                ans[0, (idx+t):(idx+t+1), :, 4*tl[0]:4*(tl[0] + length_h), 4*tl[1]: 4*(tl[1] + length_w)] += oup_list[idx].numpy()
                        else:
                            pass
            print(np.argwhere(tong<0.5)) # should be none
            # 所有帧处理完之后，求平均
            ans /= tong
            

    def test_step(self, batchdata, **kwargs):
        """
            每一次传入帧i和光流i->i+1，最后一帧为99，没有光流，将这些信息储存起来
            如果当前帧是最后一帧，则开始处理程序：
            for i in range(0,99):
                如果当前帧还有块没有处理，则挖出来一条通道进行处理，将结果填入桶中，
                直到当前帧处理完毕。
                通道长度：min(T_train, 99-i+1)

            possible kwargs:
                save_image
                save_path
                ensemble
        
            可以测试不同的分块方式
        """
        lq = batchdata['lq']  #  [B,3,h,w]
        gt = batchdata.get('gt', None)  # if not None: [B,3,4*h,4*w]
        assert len(batchdata['lq_path']) == 1  # 每个sample所带的lq_path列表长度仅为1， 即自己
        lq_paths = batchdata['lq_path'][0] # length 为batch长度
        now_start_id, clip = self.get_img_id(lq_paths[0])
        now_end_id, _ = self.get_img_id(lq_paths[-1])
        assert clip == _
        
        if now_start_id==0:
            print("first frame: {}".format(lq_paths[0]))
            self.LR_list = []
            self.HR_list = []

        # pad lq
        B ,_ ,origin_H, origin_W = lq.shape
        lq = img_multi_padding(lq, padding_multi=self.eval_cfg.multi_pad, pad_method = "edge") #  edge  constant
        self.LR_list.append(lq)  # [1,3,h,w]

        if gt is not None:
            for i in range(B):
                self.HR_list.append(gt[i:i+1, ...])

        if now_end_id == 99:
            print("start to forward all frames....")
            if self.eval_cfg.gap == 1:
                # do ensemble (8 times) or not do (1 times)
                ensemble_res = []
                self.LR_list = np.concatenate(self.LR_list, axis=0) # [100, 3,h,w]
                for item in range(1): # do not have flip
                    print("do ensemble for {}".format(item))
                    inp = ensemble_forward(self.LR_list, Type=item)
                    _, _, now_h, now_w = inp.shape
                    self.HR_G = np.zeros((1, 100, 3, 4*now_h, 4*now_w)) # 往里面进行填充
                    self.test_step_block_by_block(inp, self.HR_G, clip)
                    ensemble_res.append(ensemble_back(self.HR_G, Type=item))
                self.HR_G = sum(ensemble_res) / len(ensemble_res) # ensemble_res 结果取平均
            elif self.eval_cfg.gap == 2:
                raise NotImplementedError("not implement gap != 1 now")
                # self.HR_G_1 = test_generator_batch(F.stack(self.LR_list[::2], axis=1), netG=self.generator)
                # self.HR_G_2 = test_generator_batch(F.stack(self.LR_list[1::2], axis=1), netG=self.generator) # [B,T,C,H,W]
                # # 交叉组成HR_G
                # res = []
                # _,T1,_,_,_ = self.HR_G_1.shape
                # _,T2,_,_,_ = self.HR_G_2.shape
                # assert T1 == T2
                # for i in range(T1):
                #     res.append(self.HR_G_1[:, i, ...])
                #     res.append(self.HR_G_2[:, i, ...])
                # self.HR_G = F.stack(res, axis=1) # [B,T,C,H,W]
            else:
                raise NotImplementedError("do not support eval&test gap value")
            
            scale = self.generator.upscale_factor
            # get numpy
            self.HR_G = img_de_multi_padding(self.HR_G, origin_H=origin_H * scale, origin_W=origin_W * scale) # depad for HR_G   [B,T,C,H,W]

            if kwargs.get('save_image', False):
                print("saving images to disk ...")
                save_path = kwargs.get('save_path')
                B,T,_,_,_ = self.HR_G.shape
                assert B == 1
                assert T == 100
                for i in range(T):
                    img = tensor2img(self.HR_G[0, i, ...], min_max=(-0.5, 0.5))
                    if (i+1)%10 == 0:
                        imwrite(img, file_path=os.path.join(save_path, "partframes", f"{clip}_{str(i).zfill(8)}.png"))
                    # imwrite(img, file_path=os.path.join(save_path, "allframes", f"{clip}_{str(i).zfill(8)}.png"))
                    
        return now_end_id == 99

    def cal_for_eval(self, gathered_outputs, gathered_batchdata):
        """
            :param gathered_outputs: list of tensor, [B,C,H,W]
            :param gathered_batchdata: dict, include data
            :return: eval result
        """
        if gathered_outputs:
            crop_border = self.eval_cfg.crop_border
            assert len(self.HR_list) == 100
            res = []
            for i in range(len(self.HR_list)):
                G = tensor2img(self.HR_G[0, i, ...], min_max=(-0.5, 0.5))
                gt = tensor2img(self.HR_list[i][0], min_max=(-0.5, 0.5))
                eval_result = dict()
                for metric in self.eval_cfg.metrics:
                    eval_result[metric+"_RGB"] = self.allowed_metrics[metric](G, gt, crop_border)
                    # eval_result[metric+"_Y"] = self.allowed_metrics[metric](G_key_y, gt_y, crop_border)
                res.append(eval_result)
            return res
        else:
            return []

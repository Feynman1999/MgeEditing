import megengine.functional as F
from typing import List
from edit.utils import imwrite, tensor2img, bgr2ycbcr, img_multi_padding, img_de_multi_padding
import random
import numpy as np
import copy
import cv2

def train_batch(img1, img2, 
                gt_bboxes1, gt_bboxes2, 
                gt_labels1, gt_labels2,
                *, gm_G, netG, optim_G, loss_weight, now_iter):
    netG.train()
    with gm_G:
        pre_hm = netG.get_pre_hm(gt_bboxes1, gt_labels1)
        # imwrite(tensor2img(pre_hm[0]), file_path = "./workdirs/{}_prehm.png".format(now_iter))
        heatmap, hw, motion = netG(img2, pre_img = img1, pre_hm = pre_hm)
        losses = netG.get_loss(heatmap, hw, motion, gt_bboxes2, gt_labels2, 
                                loss_weight, pre_gt_bboxes = gt_bboxes1, 
                                pre_gt_labels = gt_labels1)
        optim_G.clear_grad()
        gm_G.backward(losses[-1])
        optim_G.step()

    return losses

def check_location(x, thr = 10):
    if x>=thr and x<=240-1-thr:
        return True
    return False

def pooling_nms(hm):
    thr = 0.27
    ksize = 5
    hmax = F.max_pool2d(hm, kernel_size=ksize, stride=1, padding=ksize // 2)
    keep = (abs(hmax - hm) < 1e-7).astype("float32")
    hm = keep * hm
    hm = (hm>thr).astype("float32") * hm
    return hm

nowiter = 0

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

def test_batch(img1, img2, pre_bboxes, *, netG, pre_labels, gap):
    """
        输入：
            给定上一帧img1，当前帧img2，以及上一帧的480*480上的预测bboxes结果
        输出：
            当前帧的480*480上的预测bboxes结果，和对应类别以及id (暂时先不考虑id)
    """
    netG.eval()
    # 根据前一帧的pre_bboxes1生成测试用的pre_hm
    if pre_bboxes is None:
        pre_hm = None
    else:
        pre_hm = netG.get_test_pre_hm(pre_bboxes)
    heatmap, hw, motion = netG(img2, pre_img = img1, pre_hm = pre_hm)
    hw = hw.numpy()
    motion = motion.numpy()
    now_bboxes = []
    now_labels = []
    # heatmap: [1,C,240,240]
    #　1.find all 1 point
    heatmap = pooling_nms(heatmap).numpy()
    # for each class , get bboxes
    b,C,h,w = heatmap.shape
    assert b==1
    need_to_deal = []
    viz_img = tensor2img(img2[0], min_max=(-0.5, 0.5)).copy()
    for cla in range(C):
        for i in range(h):
            for j in range(w):
                if heatmap[0, cla, i, j] > 0.001 and check_location(i) and check_location(j):
                    feat_w = hw[0, 0, i, j]
                    feat_h = hw[0, 1, i, j]
                    origin_w = feat_w * netG.stride
                    origin_h = feat_h * netG.stride
                    origin_center_x = netG.fm_ctr[0, 0, i, j]
                    origin_center_y = netG.fm_ctr[0, 1, i, j]
                    tl_x = int(origin_center_x - origin_w/2 + 0.5)
                    tl_y = int(origin_center_y - origin_h/2 + 0.5)
                    br_x = int(origin_center_x + origin_w/2 + 0.5)
                    br_y = int(origin_center_y + origin_h/2 + 0.5)
                    now_bboxes.append([tl_x, tl_y, br_x, br_y])
                    now_labels.append([]) # 占一个位置
                    motion_w = motion[0, 0, i, j]
                    motion_h = motion[0, 1, i, j]
                    desti_x = origin_center_x + netG.stride * motion_w
                    desti_y = origin_center_y + netG.stride * motion_h
                    """
                        根据origin_center_x和desti_x绘制变化箭头
                    """
                    cv2.arrowedLine(viz_img, (int(origin_center_x),int(origin_center_y)), (int(desti_x),int(desti_y)), (0,0,255),3,8,0,0.3)
                    # cv2.rectangle(viz_img, (tl_x, tl_y), (br_x, br_y), (0, 0, 255), 1, 8)
                    # viz_img[tl_y:br_y, tl_x:br_x, :] = 255
                    need_to_deal.append((float(heatmap[0, cla, i, j]), desti_x, desti_y, cla, len(now_bboxes)-1)) # (p, desti_w, desti_h, cla, index) 
    
    return_now_bboxes = []
    return_now_labels = []

    # cal id
    if pre_labels is None:
        # 第一帧，直接赋值id，从1开始
        for deal in need_to_deal:
            _, _, _, cla, idx = deal
            return_now_labels.append([cla, idx+1])
            return_now_bboxes.append(now_bboxes[idx])            
    else:
        assert pre_bboxes is not None
        # 对need_to_deal按照p从大到小的顺序，依次贪心匹配id
        use_flags = [False] * len(pre_labels[0]) # 是否使用
        pre_bbox = pre_bboxes[0] # [S, 4]
        assert len(pre_bbox) == len(use_flags)
        center_x = (pre_bbox[:, 0] + pre_bbox[:, 2]) / 2 # w
        center_y = (pre_bbox[:, 1] + pre_bbox[:, 3]) / 2 # h
        dis_thr = (12 if gap==1 else 20)    # 为容忍的差距上限，否则新分配id
        need_to_deal = sorted(need_to_deal, reverse=True)
        """
            some check
        """
        if abs(len(need_to_deal) - len(use_flags)) > 5:
            print("前后检测出的bbox nums: {} {}".format(len(use_flags), len(need_to_deal)))
        """
            获得新id的开始编号
        """
        max_id = 0
        for i in range(len(use_flags)):
            max_id = max(max_id, pre_labels[0][i][1])
        assert max_id > 0
        max_id += 1
        order = len(need_to_deal) - 1
        for deal in need_to_deal:
            """
                将结果写到now_labels中
            """
            _, desti_x, desti_y, cla, idx = deal
            min_dis = 10000000
            # 求所有未使用的，最小的距离
            min_idx = -1
            for i in range(len(use_flags)):
                if use_flags[i]:
                    continue
                # cal dis^2
                if ((desti_x-center_x[i])**2 + (desti_y - center_y[i])**2) < min_dis:
                    min_dis = ((desti_x-center_x[i])**2 + (desti_y - center_y[i])**2)
                    min_idx = i
            if min_idx != -1 and min_dis < dis_thr ** 2:
                # print("idx: {} find pre {}".format(idx, min_idx))
                return_now_labels.append([cla, pre_labels[0][min_idx][1]])
                return_now_bboxes.append(now_bboxes[idx])
                use_flags[min_idx] = True
            elif order <= -1:
                pass # 永远不成立
            else:
                # 分配新id
                # print("idx: {} got new".format(idx))
                return_now_labels.append([cla, max_id])
                return_now_bboxes.append(now_bboxes[idx])
                max_id += 1
            order -= 1

    # 最后，根据检测到的bboxes画带id的bboxes 
    oooid = 0
    for item in return_now_bboxes:
        cv2.rectangle(viz_img, (item[0], item[1]), (item[2], item[3]), color_dict[return_now_labels[oooid][1]], 2, 8)
        oooid+=1
    global nowiter
    # imwrite(viz_img, file_path="./viz_{}_{}.png".format(nowiter, gap))
    nowiter+=1
    
    return_now_bboxes = np.array(return_now_bboxes, dtype=np.int64)
    return_now_bboxes = [return_now_bboxes]
    return_now_labels = np.array(return_now_labels, dtype=np.int64)
    return_now_labels = [return_now_labels]
    return return_now_bboxes, return_now_labels

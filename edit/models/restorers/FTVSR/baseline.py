import os
import time
import numpy as np
import megengine.distributed as dist
import megengine as mge
import megengine.functional as F
from megengine.utils.profiler import Profiler
from .utils import get_bilinear
from tqdm import tqdm

def baseline_train_generator_batch(image, label, *, gm, netG, netloss):
    B,T,_,h,w = image.shape
    biup = get_bilinear(image)
    netG.train()
    with gm:
        # cal all flows
        frames_1 = [] # [B, 3, H, W]
        frames_2 = [] # [B, 3, H, W]
        for i in range(1, T):
            frames_1.append(image[:, i, ...])
            frames_2.append(image[:, i-1, ...])
        flows_1_0 = netG.flownet(F.concat(frames_1, axis=0), F.concat(frames_2, axis=0)) # [(T-1)*B, 2, h, w]
        frames_1 = []
        frames_2 = []
        for i in range(T-1):
            frames_1.append(image[:, i, ...])
            frames_2.append(image[:, i+1, ...])
        flows_0_1 = netG.flownet(F.concat(frames_1, axis=0), F.concat(frames_2, axis=0)) # [(T-1)*B, 2, h, w]

        output = netG(image, flows_0_1, flows_1_0)
        loss = netloss(output + biup, label)
        gm.backward(loss)
        if dist.is_distributed():
            loss = dist.functional.all_reduce_sum(loss) / dist.get_world_size()
    return loss

def test(image, netG):
    # with Profiler(topic=(1 << 6) - 1):
    B,T,_,h,w = image.shape
    assert B == 1
    assert T % 5 == 0

    B,T,_,h,w = image.shape
    biup = get_bilinear(image)
    netG.eval()

    # cal all flows
    flows_1_0 = []
    for i in range(1, T):
        flows_1_0.append(netG.flownet(image[:, i, ...], image[:, i-1, ...]))
    flows_1_0 = F.concat(flows_1_0, axis=0) # [(T-1)*B, 2, h, w]

    flows_0_1 = []
    for i in range(T-1):
        flows_0_1.append(netG.flownet(image[:, i, ...], image[:, i+1, ...]))
    flows_0_1 = F.concat(flows_0_1, axis=0) # [(T-1)*B, 2, h, w]

    output = netG.test_forward(image, flows_0_1, flows_1_0)
    res = biup + output
    return res

def baseline_test_generator_batch(image, *, netG):
    res = []
    res.append(test(image[:, 0:50, ...], netG))
    res.append(test(image[:, 50:100, ...], netG))
    res = F.concat(res, axis=1) # [1,100,c,h,w]
    return res

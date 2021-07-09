import os
import time
import numpy as np
import megengine.distributed as dist
import megengine as mge
import megengine.functional as F
from .utils import get_bilinear
from tqdm import tqdm

def shortcut_train_generator_batch(image, label, *, gm, netG, netloss):
    B,T,_,h,w = image.shape
    biup = get_bilinear(image)
    netG.train()
    with gm:
        now_hiddens = []
        # get features from rgb
        rgb = image.reshape(B*T, 3, h, w)
        rgb = netG.rgb_feature(rgb).reshape(B, T, -1, h, w)
        # append to hiddens
        for t in range(T):
            now_hiddens.append(rgb[:, t, ...]) # [B, C, h, w]
        del rgb
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

        for layer_idx in range(netG.RNN_layers):
            forward_hiddens = []
            backward_hiddens = []
            hidden = F.zeros((2*B, netG.hidden_channels, h, w))
            flow = F.zeros((2*B, 2, h, w))
            hidden = netG(F.concat([now_hiddens[0], now_hiddens[T-1]], axis=0), hidden, flow, layer_idx)
            forward_hiddens.append(hidden[0:B, ...])
            backward_hiddens.append(hidden[B:2*B, ...])
            for i in range(T-1):
                f_flow = flows_1_0[i*B:(i+1)*B, ...]
                b_flow = flows_0_1[(T-i-2)*B:(T-i-1)*B, ...]
                flow = F.concat([f_flow, b_flow], axis=0) # [2B,2,H,W]
                hidden = netG(F.concat([now_hiddens[i+1], now_hiddens[T-i-2]], axis=0), hidden, flow, layer_idx)
                forward_hiddens.append(hidden[0:B, ...])
                backward_hiddens.append(hidden[B:2*B, ...])
            # update now_hiddens
            for i in range(T):
                now_hiddens[i] += netG.aggr_forward_backward_hidden(forward_hiddens[i], backward_hiddens[T-i-1], layer_idx)
        del forward_hiddens
        del backward_hiddens
        del flows_1_0
        del flows_0_1

        # concat all hiddens on batch dim
        now_hiddens = F.concat(now_hiddens, axis=0) # [T*B,C,H,W]
        now_hiddens = netG.do_upsample(now_hiddens) # [T*B,3,4*H,4*W]
        now_hiddens = now_hiddens.reshape(T, B, 3, 4*h, 4*w)
        now_hiddens = now_hiddens.transpose(1, 0, 2, 3, 4)
        loss = netloss(now_hiddens + biup, label)
        gm.backward(loss)
        if dist.is_distributed():
            loss = dist.functional.all_reduce_sum(loss) / dist.get_world_size()
    return loss

def shortcut_test_generator_batch(image, *, netG):
    # image: [1,100,3,180,320]
    B,T,_,h,w = image.shape
    biup = get_bilinear(image)
    netG.eval()
    now_hiddens = []
    # get features from rgb
    rgb = image.reshape(B*T, 3, h, w)
    rgb = netG.rgb_feature(rgb).reshape(B, T, -1, h, w)
    # append to hiddens
    for t in range(T):
        now_hiddens.append(rgb[:, t, ...]) # [B, C, h, w]
    del rgb
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

    for layer_idx in range(netG.RNN_layers):
        forward_hiddens = []
        backward_hiddens = []
        hidden = F.zeros((2*B, netG.hidden_channels, h, w))
        flow = F.zeros((2*B, 2, h, w))
        hidden = netG(F.concat([now_hiddens[0], now_hiddens[T-1]], axis=0), hidden, flow, layer_idx)
        forward_hiddens.append(hidden[0:B, ...])
        backward_hiddens.append(hidden[B:2*B, ...])
        for i in range(T-1):
            f_flow = flows_1_0[i*B:(i+1)*B, ...]
            b_flow = flows_0_1[(T-i-2)*B:(T-i-1)*B, ...]
            flow = F.concat([f_flow, b_flow], axis=0) # [2B,2,H,W]
            hidden = netG(F.concat([now_hiddens[i+1], now_hiddens[T-i-2]], axis=0), hidden, flow, layer_idx)
            forward_hiddens.append(hidden[0:B, ...])
            backward_hiddens.append(hidden[B:2*B, ...])
        # update now_hiddens
        for i in range(T):
            now_hiddens[i] += netG.aggr_forward_backward_hidden(forward_hiddens[i], backward_hiddens[T-i-1], layer_idx)
    del forward_hiddens
    del backward_hiddens
    del flows_1_0
    del flows_0_1

    # concat all hiddens on batch dim
    # now_hiddens = F.concat(now_hiddens, axis=0) # [T*B,C,H,W]
    # now_hiddens = netG.do_upsample(now_hiddens) # [T*B,3,4*H,4*W]
    # now_hiddens = now_hiddens.reshape(T, B, 3, 4*h, 4*w)
    # now_hiddens = now_hiddens.transpose(1, 0, 2, 3, 4)
    res = []
    for i in tqdm(range(T)):
        res.append(netG.do_upsample(now_hiddens[i]))
    res = F.stack(res, axis = 1) # [B,T,3,H,W]
    return res + biup

"""
    test your model.
"""
import os
# os.environ['MGB_CUDA_RESERVE_MEMORY'] = '1'
import sys
from collections.abc import Sequence
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import multiprocessing as mp
import time
import argparse
import megengine as mge
import megengine.distributed as dist
from megengine.jit import trace
from megengine.data import RandomSampler, SequentialSampler, DataLoader
import os.path as osp
from megengine.data.dataset import Dataset

import logging
from megengine.distributed.group import get_rank
from megengine.distributed import is_distributed
import inspect

from enum import Enum


class Priority(Enum):
    """Hook priority levels.

    +------------+------------+
    | Level      | Value      |
    +============+============+
    | HIGHEST    | 0          |
    +------------+------------+
    | VERY_HIGH  | 10         |
    +------------+------------+
    | HIGH       | 30         |
    +------------+------------+
    | NORMAL     | 50         |
    +------------+------------+
    | LOW        | 70         |
    +------------+------------+
    | VERY_LOW   | 90         |
    +------------+------------+
    | LOWEST     | 100        |
    +------------+------------+
    """

    HIGHEST = 0
    VERY_HIGH = 10
    HIGH = 30
    NORMAL = 50
    LOW = 70
    VERY_LOW = 90
    LOWEST = 100


def get_priority(priority):
    """Get priority value.

    Args:
        priority (int or str or :obj:`Priority`): Priority.

    Returns:
        int: The priority value.
    """
    if isinstance(priority, int):
        if priority < 0 or priority > 100:
            raise ValueError('priority must be between 0 and 100')
        return priority
    elif isinstance(priority, Priority):
        return priority.value
    elif isinstance(priority, str):
        return Priority[priority.upper()].value
    else:
        raise TypeError('priority must be an integer or Priority enum value')

load_from = './workdirs/centertracker_fish/20210606_003217/checkpoints/epoch_100'
dataroot = "/data/home/songtt/chenyuxiang/datasets/MOTFISH/preliminary/test"
exp_name = 'centertracker_fish_test'
input_h = 480
input_w = 480
loss_weight = {
    "hms": 1,
    "hw": 1,
    "motion": 1,
}
fp = 0.1 # 在周围随机再加一个框的概率 默认0.1
fn = 0.4 # 去除一个框的概率 默认0.4

import copy
from abc import ABCMeta, abstractmethod
from megengine.data.dataset import Dataset
import megengine.module as M
import os
import time
from megengine.functional.tensor import concat
import numpy as np
import cv2
import megengine.distributed as dist
import megengine as mge
import megengine.functional as F
from megengine.autodiff import GradManager

from tqdm import tqdm
from collections import defaultdict

import random
import pandas as pd
import cv2
import numpy as np
import megengine
import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
from megengine import Tensor


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

def default_init_weights(module, scale=1, nonlinearity="relu", lrelu_value = 0.1):
    """
        nonlinearity: leaky_relu
    """
    for m in module.modules():
        if isinstance(m, M.Conv2d):
            M.init.msra_normal_(m.weight, a=lrelu_value, mode="fan_in", nonlinearity=nonlinearity)
            m.weight *= scale
            if m.bias is not None:
                M.init.zeros_(m.bias)
        elif isinstance(m, M.ConvTranspose2d):
            M.init.normal_(m.weight, 0, 0.001)
            m.weight *= scale
            if m.bias is not None:
                M.init.zeros_(m.bias)
        else:
            pass


class BasicBlock(M.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = M.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn1 = M.GroupNorm(8, planes)
        self.relu = M.ReLU()
        self.conv2 = M.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = M.GroupNorm(8, planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(M.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = M.Conv2d(inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = M.GroupNorm(8, bottle_planes)
        self.conv2 = M.Conv2d(bottle_planes, bottle_planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = M.GroupNorm(8, bottle_planes)
        self.conv3 = M.Conv2d(bottle_planes, planes, kernel_size=1, bias=False)
        self.bn3 = M.GroupNorm(8, planes)
        self.relu = M.ReLU()
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)
        return out


class Root(M.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = M.Conv2d(in_channels, out_channels, 1, stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = M.GroupNorm(8, out_channels)
        self.relu = M.ReLU()
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(F.concat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)
        return x

class Tree(M.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1, level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = M.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = M.Sequential(
                M.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                M.GroupNorm(8, out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x

class DLA(M.Module):
    def __init__(self, levels, channels, num_classes=1, block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes

        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1])
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2, level_root=False, root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2, level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2, level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2, level_root=True, root_residual=residual_root)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                M.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                M.GroupNorm(8, planes),
                M.ReLU()])
            inplanes = planes
        return M.Sequential(*modules)

    def forward(self, x):
        y = []
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        # 此处相对于dla34的返回结果进行了变动，返回6个level的特征图
        return y


def dla34(**kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [32, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    return model


def fill_up_weights(up):
    w = up.weight
    # print(w.shape) #   (groups, in_channels // groups, out_channels // groups, height, width)
    f = math.ceil(w.shape[3] / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.shape[3]):
        for j in range(w.shape[4]):
            w[0, 0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.shape[0]):
        w[c, 0, 0, :, :] = w[0, 0, 0, :, :]


class DeformConv(M.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = M.Sequential(
            M.GroupNorm(8, cho),
            M.ReLU()
        )
        channels_ = 3 * 3 * 3
        self.conv_offset_mask = M.Conv2d(chi,
                                    channels_,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=True)
        self.deformconv = M.DeformableConv2d(chi, cho, kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, x):
        # get offset and mask
        out = self.conv_offset_mask(x)
        # out: b,27,h,w
        offset = out[:, 0:18, :, :]
        mask = out[:, 18:, :, :]
        mask = F.sigmoid(mask)
        x = self.deformconv(x, offset, mask)
        x = self.actf(x)
        return x


class IDAUp(M.Module):
    '''
    IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j])
    ida(layers, len(layers) -i - 2, len(layers))
    '''
    def __init__(self, o, channels, up_f):
        # j = -3
        # o: 128 
        # channels: [128,256,512]
        # up_f: [2/2, 4/2,8/2] = [1,2,4]
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)): # i in 1, 2
            c = channels[i]
            f = int(up_f[i])  # 2 
            proj = DeformConv(c, o)
            node = DeformConv(o, o)

            up = M.ConvTranspose2d(o, o, f *2, stride=f, padding=f//2, groups=1, bias=False)
            # fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
                 
    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):# i in 4, 5
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))

            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(M.Module):
    '''
    # first_level = 2 if down_ratio=4
    # channels = [64, 128, 256, 512]
    # scales = [1, 2, 4, 8]
    '''
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)

        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i), IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class DLASeg_GN(M.Module):
    def __init__(self, num_layers, inp, pretrained):
        super(DLASeg_GN, self).__init__()
        self.first_level = 2
        self.last_level = 5

        self.base = dla34()
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))] # [1,2,4,8]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)
        out_channel = channels[self.first_level]

        # 进行上采样
        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], 
                            [2 ** i for i in range(self.last_level - self.first_level)])
        self.init_weights()

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i])

        self.ida_up(y, 0, len(y))

        return y[-1]

    def init_weights(self):
        default_init_weights(self, scale=0.2)


def gaussian2D(radius, sigma=1):
    """Generate 2D gaussian kernel.
    Args:
        radius (int): Radius of gaussian kernel.
        sigma (int): Sigma of gaussian function. Default: 1.
    Returns:
        h (Tensor): Gaussian kernel with a
            ``(2 * radius + 1) * (2 * radius + 1)`` shape.
    """
    x = F.arange(-radius, radius + 1).reshape(1, -1)
    y = F.arange(-radius, radius + 1).reshape(-1, 1)
    h = F.exp((-(x * x + y * y) / (2 * sigma * sigma)))
    h[h < 1.1920928955e-07 * h.max()] = 0.0
    return h

def gen_gaussian_target(heatmap, center, radius, k=1):
    """Generate 2D gaussian heatmap.
    Args:
        heatmap (Tensor): Input heatmap, the gaussian kernel will cover on
            it and maintain the max value.
        center (list[int]): Coord of gaussian kernel's center.
        radius (int): Radius of gaussian kernel.
        k (int): Coefficient of gaussian kernel. Default: 1.
    Returns:
        out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
    """
    diameter = 2 * radius + 1
    gaussian_kernel = gaussian2D(radius, sigma=diameter / 6)
    height, width = heatmap.shape[:2]

    x, y = center
    x = max(0, x)
    y = max(0, y)
    x = min(width-1, x)
    y = min(height-1, y)
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_gaussian = gaussian_kernel[radius - top:radius + bottom, radius - left:radius + right]
    assert heatmap.dtype == masked_gaussian.dtype
    heatmap[y - top:y + bottom, x - left:x + right] = F.maximum(heatmap[y - top:y + bottom, x - left:x + right], masked_gaussian * k)
    return heatmap

def safelog(x, eps=None):
    if eps is None:
        eps = np.finfo(x.dtype).eps
    return F.log(F.maximum(x, eps))

class Center_loss(M.Module):
    def __init__(self, alpha = 2, beta = 4):
        super(Center_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, label):
        """
            pred: [B,C,H,W]  C: class nums  H, W: score map size
            label: same to pred, have dealed by gauss function 
        """
        # first cal N (objects) nums
        # 对于每个batch，统计N
        b,c,h,w = label.shape
        peak = (label > 0.99999).astype("float32") # [b,c,h,w]
        peak = peak.reshape(b, -1) # [b, c*h*w]
        pred = pred.reshape(b, -1)
        label = label.reshape(b, -1)
        
        # weight for each sample
        N = F.sum(peak, axis=1, keepdims=True) # [b, 1] 每个batch中物体的数量 
        
        assert float(F.min(N).item()) > 0.000001,  "should contain at least one object in a sample"

        # peak loss
        peak_loss = ((peak * ((1-pred)**self.alpha) * safelog(pred))) # [b, -1]

        # not peak loss
        not_peak_loss = ((1-peak) *  ((1-label)**self.beta) * (pred**self.alpha) * safelog(1-pred)) # [b,-1]

        return -((peak_loss + not_peak_loss) / N).sum() / b

def is_var(tensor):
    return isinstance(tensor, Tensor)


def is_ndarray(tensor):
    return isinstance(tensor, np.ndarray)


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """Convert variable or ndarray into image numpy arrays.

    After clamping to (min, max), image values will be normalized to [0, 1].

    For differnet tensor shapes, this function will have different behaviors:

        1. 4D mini-batch Tensor of shape (N x 3/1 x H x W):
            Use `make_grid` to stitch images in the batch dimension, and then
            convert it to numpy array.
        2. 3D Tensor of shape (3/1 x H x W) and 2D Tensor of shape (H x W):
            Directly change to numpy array.

    Note that the image channel in input tensors should be RGB order. This
    function will convert it to cv2 convention, i.e., (H x W x C) with BGR
    order.

    Args:
        tensor (Tensor | list[Tensor]): Input tensors.
        out_type (numpy type): Output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple): min and max values for clamp.

    Returns:
        (Tensor | list[Tensor]): 3D ndarray of shape (H x W x C) or 2D ndarray
        of shape (H x W).
    """
    if is_var(tensor):
        tensor = tensor.to("cpu0").astype('float32').numpy()
    elif isinstance(tensor, list) and all(is_var(t) for t in tensor):
        tensor = [t.to("cpu0").astype('float32').numpy() for t in tensor]
    else:
        assert is_ndarray(tensor) or (isinstance(tensor, list) and all(is_ndarray(t) for t in tensor))

    if is_ndarray(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        # Squeeze two times so that:
        # 1. (1, 1, h, w) -> (h, w) or
        # 3. (1, 3, h, w) -> (3, h, w) or
        # 2. (n>1, 3/1, h, w) -> (n>1, 3/1, h, w)
        _tensor = np.squeeze(_tensor)
        _tensor = np.clip(_tensor, a_min=min_max[0], a_max=min_max[1])
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = len(_tensor.shape)
        if n_dim == 4:
            raise NotImplementedError("dose not support mini batch var2image")
            # img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            # img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
        elif n_dim == 3:
            img_np = np.transpose(_tensor[[2, 1, 0], :, :], (1, 2, 0))  # CHW -> HWC and rgb -> bgr
        elif n_dim == 2:
            img_np = _tensor
        else:
            raise ValueError('Only support 4D, 3D or 2D tensor. '
                             f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        result.append(img_np.astype(out_type))
    result = result[0] if len(result) == 1 else result
    return result

def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = osp.abspath(osp.dirname(file_path))
        mkdir_or_exist(dir_name)
    return cv2.imwrite(file_path, img, params)

from pathlib import Path

def scandir(dir_path, suffix=None, recursive=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str | obj:`Path`): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                if suffix is None:
                    yield rel_path
                elif rel_path.endswith(suffix):
                    yield rel_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

backbone_spec = {'DLA_GN': DLASeg_GN}

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
                    cv2.arrowedLine(viz_img, (int(origin_center_x),int(origin_center_y)), (int(desti_x),int(desti_y)), (0,0,255),5,8,0,0.3)
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
        cv2.rectangle(viz_img, (item[0], item[1]), (item[2], item[3]), color_dict[return_now_labels[oooid][1]], 1, 8)
        oooid+=1
    global nowiter
    imwrite(viz_img, file_path="./viz_{}_{}.png".format(nowiter, gap))
    nowiter+=1
    
    return_now_bboxes = np.array(return_now_bboxes, dtype=np.int64)
    return_now_bboxes = [return_now_bboxes]
    return_now_labels = np.array(return_now_labels, dtype=np.int64)
    return_now_labels = [return_now_labels]
    return return_now_bboxes, return_now_labels

epoch_dict = {}

def adjust_learning_rate(optimizer, epoch):
    if epoch>=80 and epoch % 10 == 0  and epoch_dict.get(epoch, None) is None:
        epoch_dict[epoch] = True
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * 0.3
        print("adjust lr! , now lr: {}".format(param_group["lr"]))

class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for Dataset.

    All datasets should subclass it.
    All subclasses should overwrite:

    ``load_annotations``, supporting to load information and generate image lists.

    Args:
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (bool): If True, the dataset will work in test mode. Otherwise, in train mode.
    """
    def __init__(self, pipeline, mode='train'):
        super(BaseDataset, self).__init__()
        assert mode in ("train", "test", "eval")
        self.mode = mode
        self.pipeline = Compose(pipeline)
        self.logger = get_root_logger()

    @abstractmethod
    def load_annotations(self):
        """Abstract function for loading annotation.

        All subclasses should overwrite this function
        """

    @abstractmethod
    def evaluate(self, results):
        """Abstract function for evaluate.

        All subclasses should overwrite this function
        """

    def prepare_data(self, idx):
        """Prepare training data.

        Args:
            idx (int): Index of the training batch data.

        Returns:
            dict: Returned training batch.
        """
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        """Length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data_infos)

    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """
        return self.prepare_data(idx)

class Registry:
    """A registry to map strings to classes.

    Args:
        name (str): Registry name.
    """

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + \
                     f'(name={self._name}, ' \
                     f'items={self._module_dict})'
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        """Get the registry record.

        Args:
            key (str): The class name in string format.

        Returns:
            class: The corresponding class.
        """
        return self._module_dict.get(key, None)

    def _register_module(self, module_class, module_name=None, force=False):
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, '
                            f'but got {type(module_class)}')

        if module_name is None:
            module_name = module_class.__name__
        if not force and module_name in self._module_dict:
            raise KeyError(f'{module_name} is already registered '
                           f'in {self.name}')
        self._module_dict[module_name] = module_class

    def register_module(self, name=None, force=False, module=None):
        """Register a module.

        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.

        Example:
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(name = "ResNet", module = ResNet)

        Args:
            name (str | None): The module name to be registered. If not
                specified, the class name will be used.
            force (bool, optional): Whether to override an existing class with
                the same name. Default: False.
            module (type): Module class to be registered.
        """
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)):
            raise TypeError(f'name must be a str, but got {type(name)}')

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module_class=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(cls):
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls

        return _register

MODELS = Registry('model')
BACKBONES = Registry('backbone')
COMPONENTS = Registry('component')
LOSSES = Registry('loss')
DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
HOOKS = Registry('hook')

class BaseStorageBackend(metaclass=ABCMeta):
    """Abstract class of storage backends.

    All backends need to implement two apis: ``get()`` and ``get_text()``.
    ``get()`` reads the file as a byte stream and ``get_text()`` reads the file
    as texts.
    """

    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass

class HardDiskBackend(BaseStorageBackend):
    """Raw hard disks storage backend."""

    def get(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf

    def get_text(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'r') as f:
            value_buf = f.read()
        return value_buf

class FileClient:
    """A general file client to access files in different backend.

    The client loads a file or text in a specified backend from its path
    and return it as a binary file. it can also register other backend
    accessor with a given name and backend class.

    Attributes:
        backend (str): The storage backend type. Options are "disk", "ceph",
            "memcached" and "lmdb".
        client (:obj:`BaseStorageBackend`): The backend object.
    """

    _backends = {
        'disk': HardDiskBackend,
    }

    def __init__(self, backend='disk', **kwargs):
        if backend not in self._backends:
            raise ValueError(
                f'Backend {backend} is not supported. Currently supported ones'
                f' are {list(self._backends.keys())}')
        self.backend = backend
        self.client = self._backends[backend](**kwargs)

    @classmethod
    def _register_backend(cls, name, backend, force=False):
        if not isinstance(name, str):
            raise TypeError('the backend name should be a string, '
                            f'but got {type(name)}')
        if not inspect.isclass(backend):
            raise TypeError(
                f'backend should be a class but got {type(backend)}')
        if not issubclass(backend, BaseStorageBackend):
            raise TypeError(
                f'backend {backend} is not a subclass of BaseStorageBackend')
        if not force and name in cls._backends:
            raise KeyError(
                f'{name} is already registered as a storage backend, '
                'add "force=True" if you want to override it')

        cls._backends[name] = backend

    @classmethod
    def register_backend(cls, name, backend=None, force=False):
        """Register a backend to FileClient.

        This method can be used as a normal class method or a decorator.

        .. code-block:: python

            class NewBackend(BaseStorageBackend):

                def get(self, filepath):
                    return filepath

                def get_text(self, filepath):
                    return filepath

            FileClient.register_backend('new', NewBackend)

        or

        .. code-block:: python

            @FileClient.register_backend('new')
            class NewBackend(BaseStorageBackend):

                def get(self, filepath):
                    return filepath

                def get_text(self, filepath):
                    return filepath

        Args:
            name (str): The name of the registered backend.
            backend (class, optional): The backend class to be registered,
                which must be a subclass of :class:`BaseStorageBackend`.
                When this method is used as a decorator, backend is None.
                Defaults to None.
            force (bool, optional): Whether to override the backend if the name
                has already been registered. Defaults to False.
        """
        if backend is not None:
            cls._register_backend(name, backend, force=force)
            return

        def _register(backend_cls):
            cls._register_backend(name, backend_cls, force=force)
            return backend_cls

        return _register

    def get(self, filepath):
        return self.client.get(filepath)

    def get_text(self, filepath):
        return self.client.get_text(filepath)

def imfrombytes(content, flag='color', channel_order='bgr', backend=None):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Same as :func:`imread`.
        backend (str|None): The image decoding backend type. Options are `cv2`,
            `pillow`, `turbojpeg`, `None`. If backend is None, the global
            imread_backend specified by ``use_backend()`` will be used.
            Default: None.

    Returns:
        ndarray: Loaded image array.
    """
    imread_backend = 'cv2'
    if backend is None:
        backend = imread_backend
    from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED
    imread_flags = {
        'color': IMREAD_COLOR,
        'grayscale': IMREAD_GRAYSCALE,
        'unchanged': IMREAD_UNCHANGED
    }

    img_np = np.frombuffer(content, np.uint8)
    flag = imread_flags[flag] if isinstance(flag, str) else flag
    img = cv2.imdecode(img_np, flag)
    if flag == IMREAD_COLOR and channel_order == 'rgb':
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    return img

from megengine.data.transform import Resize as mge_resize

interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


def contrast(image, value):
    if value <= 0:
        return image
    dtype = image.dtype
    image = image.astype(np.float32)
    alpha = value
    image = image * alpha + image.mean() * (1 - alpha)
    return image.clip(0, 255).astype(dtype)

@PIPELINES.register_module()
class Add_contrast(object):
    def __init__(self, keys, value):
        self.keys = keys
        self.value = value

    def __call__(self, results):
        for key in self.keys:
            if isinstance(results[key], list):
                raise NotImplementedError("not support list key")
            else:
                if key in ['img', ]:
                    results[key] = contrast(results[key], value = self.value)
                else:
                    raise NotImplementedError("not support key")
        return results

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string

@PIPELINES.register_module()
class Resize(object):
    """
        Args:
        size (int|list|tuple): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int): Interpolation mode of resize. Default: cv2.INTER_LINEAR.
    """
    def __init__(self, keys, size, interpolation='bilinear'):
        assert interpolation in interp_codes
        self.keys = keys
        self.size = size
        self.interpolation_str = interpolation
        self.resize = mge_resize(output_size=self.size, interpolation=interp_codes[interpolation])

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            if isinstance(results[key], list):
                results[key] = [
                    self.resize.apply(v) for v in results[key]
                ]
            else:
                """
                    当key不是list时，会检测有没有bbox，如果有则进行bbox的调整
                    先记录下每个key之前的shape
                """
                if 'scale_factor' not in results.keys():
                    old_h, old_w, _ = results[key].shape
                    results['scale_factor'] = np.array([self.size[1]/old_w, self.size[0]/old_h, self.size[1]/old_w, self.size[0]/old_h])
                    self._resize_bboxes(results)
                results[key] = self.resize.apply(results[key])
        return results

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        num = 0
        for key in results.keys():
            if "bbox" in key:
                num += 1
                bboxes = np.around(results[key] * results['scale_factor'])
                results[key] = bboxes
        assert num <= 2

    def __repr__(self):
        interpolate_str = self.interpolation_str
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


@PIPELINES.register_module()
class RescaleToZeroOne(object):
    """Transform the images into a range between 0 and 1.

    Required keys are the keys in attribute "keys", added or modified keys are
    the keys in attribute "keys".
    It also supports rescaling a list of images.

    Args:
        keys (Sequence[str]): The images to be transformed.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            if isinstance(results[key], list):
                results[key] = [
                    v.astype(np.float32) / 255. for v in results[key]
                ]
            else:
                results[key] = results[key].astype(np.float32) / 255.
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class Collect(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "gt_labels".

    Args:
        keys (Sequence[str]): Required keys to be collected.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        data = {}
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + (
            f'(keys={self.keys})')


@PIPELINES.register_module()
class ImageToTensor(object):
    """
    [HWC] -> [CHW]

    Args:
        keys (Sequence[str]): Required keys to be converted.
        to_float32 (bool): Whether convert numpy image array to np.float32
            before converted to tensor. Default: True.
    """
    def __init__(self, keys, to_float32=True, do_not_stack = False):
        self.keys = keys
        self.to_float32 = to_float32
        self.do_not_stack = do_not_stack

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            # deal with gray scale img: expand a color channel
            if len(results[key].shape) == 2:
                results[key] = results[key][..., None]
            if self.to_float32 and not isinstance(results[key], np.float32):
                results[key] = results[key].astype(np.float32)
            results[key] = results[key].transpose(2, 0, 1)  # [HWC] -> [CHW]
        return results

    def __repr__(self):
        return self.__class__.__name__ + (
            f'(keys={self.keys}, to_float32={self.to_float32})')



def imnormalize_(img, mean, std, to_rgb=True):
    """Inplace normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img


def imnormalize(img, mean, std, to_rgb=True):
    """Normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    img = img.copy().astype(np.float32)
    return imnormalize_(img, mean, std, to_rgb)

@PIPELINES.register_module()
class Normalize(object):
    """Normalize images with the given mean and std value.

    Required keys are the keys in attribute "keys", added or modified keys are
    the keys in attribute "keys" and these keys with postfix '_norm_cfg'.
    It also supports normalizing a list of images.

    Args:
        keys (Sequence[str]): The images to be normalized.
        mean (np.ndarray): Mean values of different channels.
        std (np.ndarray): Std values of different channels.
        to_rgb (bool): Whether to convert channels from BGR to RGB.
    """

    def __init__(self, keys, mean, std, to_rgb=False):
        self.keys = keys
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            if isinstance(results[key], list):
                results[key] = [
                    imnormalize(v, self.mean, self.std, self.to_rgb)
                    for v in results[key]
                ]
            else:
                results[key] = imnormalize(results[key], self.mean, self.std, self.to_rgb)

        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, mean={self.mean}, std={self.std}, '
                     f'to_rgb={self.to_rgb})')

        return repr_str


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load image from file.

    Args:
        io_backend (str): io backend where images are store. Default: 'disk'.
        key (str): Keys in results to find corresponding path. Default: 'gt'.
        flag (str): Loading flag for images. Default: 'color'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
        save_original_img (bool): If True, maintain a copy of the image in
            `results` dict with name of `f'ori_{key}'`. Default: False.
        kwargs (dict): Args for file client.
    """

    def __init__(self,
                 io_backend='disk',
                 key='gt',
                 flag='color',
                 channel_order='bgr',
                 save_original_img=False,
                 make_bin=False,
                 **kwargs):
        self.io_backend = io_backend
        self.key = key
        self.flag = flag
        self.save_original_img = save_original_img
        self.channel_order = channel_order
        self.kwargs = kwargs
        self.make_bin = make_bin # 注意使用make_bin之前请先使用单gpu 单进程模式跑一个epoch，确保所有文件都已经创建bin
        self.file_client = None

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        filepath = str(results[f'{self.key}_path'])
        img_bytes = self.file_client.get(filepath)
        img = imfrombytes(img_bytes, flag=self.flag, channel_order=self.channel_order)  # HWC
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        results[self.key] = img
        results[f'{self.key}_path'] = filepath
        results[f'{self.key}_ori_shape'] = img.shape
        if self.save_original_img:
            results[f'ori_{self.key}'] = img.copy()
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(io_backend={self.io_backend}, key={self.key}, '
            f'flag={self.flag}, save_original_img={self.save_original_img})')
        return repr_str

class BaseMotDataset(BaseDataset):
    def __init__(self, pipeline, mode="train"):
        super(BaseMotDataset, self).__init__(pipeline, mode)
    
    def evaluate(self, results, save_path):
        """ Evaluate with different metrics.
            Args:
                results (list of dict): for every dict, record metric -> value for one frame

            Return:
                dict: Evaluation results dict.
        """
        # average the results
        eval_results = {}
        return eval_results
    
IMG_EXTENSIONS = ('.PNG', '.png')

@DATASETS.register_module()
class MotFishTestDataset(BaseMotDataset):
    def __init__(self,
                 folder,
                 pipeline):
        super(MotFishTestDataset, self).__init__(pipeline, "test")
        self.folder = str(folder)
        self.data_infos = self.load_annotations()
        self.logger.info("MotFishTestDataset dataset load ok,   mode: {}   len:{}".format(self.mode, len(self.data_infos)))

    def add_infos(self, gap, infos, imgs, clipname):
        Len = 0
        for i in range(0, len(imgs), gap):
            Len += 1
        idx = 0
        for i in range(0, len(imgs), gap):
            infos.append(
                dict(
                    img_path = os.path.join(self.folder, clipname, "img1", imgs[i]),
                    total_len = Len,
                    index = idx,
                    clipname= clipname,
                    gap = gap
                )
            )
            idx+=1

    def load_annotations(self):
        data_infos = []
        for clipname in os.listdir(self.folder):
            imgs = sorted(list(scandir(os.path.join(self.folder, clipname, "img1"), suffix=IMG_EXTENSIONS, recursive=False)))
            self.add_infos(gap = 1, infos = data_infos, imgs = imgs, clipname = clipname)
            self.add_infos(gap = 5, infos = data_infos, imgs = imgs, clipname = clipname)
        return data_infos

class Hook:

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass

    def before_train_epoch(self, runner):
        self.before_epoch(runner)

    def before_test_epoch(self, runner):
        self.before_epoch(runner)

    def after_train_epoch(self, runner):
        self.after_epoch(runner)

    def after_test_epoch(self, runner):
        self.after_epoch(runner)

    def before_train_iter(self, runner):
        self.before_iter(runner)

    def before_test_iter(self, runner):
        self.before_iter(runner)

    def after_train_iter(self, runner):
        self.after_iter(runner)

    def after_test_iter(self, runner):
        self.after_iter(runner)

    def every_n_epochs(self, runner, n):
        return (runner.epoch + 1) % n == 0 if n > 0 else False

    def every_n_inner_iters(self, runner, n):
        return (runner.inner_iter + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, runner, n):
        return (runner.iter + 1) % n == 0 if n > 0 else False

    def end_of_epoch(self, runner):
        return runner.inner_iter + 1 == len(runner.data_loader)

module_ckpt_suffix = "_module.mge"
optim_ckpt_suffix = "_optim.mge"

class BaseRunner(metaclass=ABCMeta):
    """The base class of Runner, a training helper for Mge.

    All subclasses should implement the following APIs:

    - ``run()``
    - ``train()``
    - ``test()``
    - ``save_checkpoint()``
    - ``resume()``

    Args:
        model (:obj:`megengine.module.Module`): The model to be run.
        optimizers_cfg (dict): optimizer configs
        work_dir (str, optional): The working directory to save checkpoints and logs. Defaults to None.
    """

    def __init__(self, model, optimizers_cfg=None, work_dir=None):
        assert hasattr(model, 'train_step')
        assert hasattr(model, 'test_step')
        # assert hasattr(model, 'create_gradmanager_and_optimizers')
        assert hasattr(model, 'cal_for_eval')

        self.model = model
        self.optimizers_cfg = optimizers_cfg
        self.logger = get_root_logger()
        self.work_dir = work_dir
        assert self.work_dir is not None

        # get model name from the model class
        self._model_name = self.model.__class__.__name__
        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    @abstractmethod
    def train(self, data_loader):
        pass

    @abstractmethod
    def test(self, data_loader):
        pass

    @abstractmethod
    def run(self, data_loaders, workflow, max_iters):
        pass

    @abstractmethod
    def save_checkpoint(self, out_dir, create_symlink=True):
        pass

    @abstractmethod
    def resume(self, path2checkpoint):
        pass

    @abstractmethod
    def register_training_hooks(self, checkpoint_config, log_config):
        """Register default hooks for training.

            Default hooks include:

            - CheckpointSaverHook
            - log_config
        """
        pass

    def create_gradmanager_and_optimizers(self):
        self.model.create_gradmanager_and_optimizers(self.optimizers_cfg)

    def sync_model_params(self):
        if dist.is_distributed():
            self.logger.info("syncing the model's parameters...")
            dist.bcast_list_(self.model.parameters(), dist.WORLD)
            dist.bcast_list_(self.model.buffers(), dist.WORLD)
        else:
            pass  # do nothing

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
                param groups. If the runner has a dict of optimizers, this
                method will return a dict.
        """
        raise NotImplementedError("")
        # if isinstance(self.optimizer, Optimizer):
        #     lr = [group['lr'] for group in self.optimizer.param_groups]
        # elif isinstance(self.optimizer, dict):
        #     lr = dict()
        #     for name, optim in self.optimizer.items():
        #         lr[name] = [group['lr'] for group in optim.param_groups]
        # else:
        #     raise RuntimeError('lr is not applicable because optimizer does not exist.')
        # return lr

    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.

        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hook')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def call_hook(self, fn_name):
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def load_checkpoint(self, path2checkpoint, load_optim=True):
        """
            :param path2checkpoint: e.g. workdirs/xxxxx/checkpoint/epoch_10
            :return: dict
        """
        assert osp.exists(path2checkpoint), "{} do not exist".format(path2checkpoint)
        dirname = osp.split(path2checkpoint)[-1]
        epoch, nums = dirname.split("_")
        assert epoch in ("epoch", )
        self.logger.info('load checkpoint from {}'.format(path2checkpoint))
        # 遍历model中的所有配置optimizer的model，并进行load
        res = dict()
        res['nums'] = int(nums)
        for submodule_name in self.optimizers_cfg.keys():
            submodule = getattr(self.model, submodule_name, None)
            assert submodule is not None, "model should have submodule {}".format(submodule_name)
            assert isinstance(submodule, M.Module), "submodule should be instance of mge.module.Module"
            if dist.get_rank() == 0:
                module_state_dict = mge.load(osp.join(path2checkpoint, submodule_name + module_ckpt_suffix))
                submodule.load_state_dict(module_state_dict, strict = False)
            if load_optim:
                optim_state_dict = mge.load(osp.join(path2checkpoint, submodule_name + optim_ckpt_suffix))
                res[submodule_name] = optim_state_dict
        return res

    def register_checkpoint_hook(self, checkpoint_config):
        if isinstance(checkpoint_config, dict):
            checkpoint_config.setdefault('type', 'CheckpointHook')
            hook = build_from_cfg(checkpoint_config, HOOKS)
        else:
            hook = checkpoint_config
        self.register_hook(hook)

    def register_logger_hooks(self, log_config):
        log_interval = log_config['interval']
        for info in log_config['hooks']:
            logger_hook = build_from_cfg(info, HOOKS, default_args=dict(interval=log_interval))
            self.register_hook(logger_hook, priority='HIGH')

class EpochBasedRunner(BaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def train(self, data_loader):
        self.mode = 'train'
        self.data_loader = data_loader
        self.call_hook('before_train_epoch')
        time.sleep(0.05)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.losses = self.model.train_step(data_batch, self._epoch, self._iter)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def test(self, data_loader):
        self.mode = 'test'
        self.data_loader = data_loader
        self.call_hook('before_test_epoch')
        time.sleep(0.05)
        save_path = self.work_dir
        mkdir_or_exist(save_path)
        for i, data_batch in enumerate(data_loader):
            batchdata = data_batch
            self._inner_iter = i
            self.call_hook('before_test_iter')
            self.outputs = self.model.test_step(batchdata, 
                                                save_image = True, 
                                                save_path = save_path)
            self.call_hook('after_test_iter')
            self._iter += 1

        self.call_hook('after_test_epoch')
        self._epoch += 1

    def run(self, data_loaders, workflow, max_epochs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training and test.
            workflow : train or test
            max_epochs (int): Total training epochs.
        """
        assert isinstance(data_loaders, list)
        assert workflow in ('test', 'train')
        assert len(data_loaders) == 1, "only support just length one data_loaders now"

        self._max_epochs = max_epochs
        if workflow == 'train':
            self._max_iters = self._max_epochs * len(data_loaders[0])
            self._iter = self.epoch * len(data_loaders[0])
            self.logger.info("{} iters for one epoch, trained iters: {}, total iters: {}".format(len(data_loaders[0]), self._iter, self._max_iters))
            # set the epoch in the dist sampler, so that the data is consitent
            # data_loaders[0].sampler.epoch = self.epoch , do not like torch and paddle, it's index are generated by np.random.RandomState(seed)
        else:
            assert max_epochs in (1, 4, 8)

        self.logger.info("Start running, work_dir: {}, workflow: {}, max epochs : {}".format(self.work_dir, workflow, max_epochs))
        self.logger.info("registered hooks: " + str(self.hooks))

        self.call_hook('before_run')
        while self.epoch < max_epochs:
            if isinstance(workflow, str):  # self.train()
                if not hasattr(self, workflow):
                    raise ValueError(f'runner has no method named "{workflow}" to run an epoch')
                epoch_runner = getattr(self, workflow)
            else:
                raise TypeError('mode in workflow must be a str, but got {}'.format(type(workflow)))
            epoch_runner(data_loaders[0])

        time.sleep(0.05)  # wait for some hooks like loggers to finish

        # if workflow == 'test':
        #     save_path = osp.join(self.work_dir, "test_results")
        #     data_loaders[0].dataset.test_aggre(save_path)  # for video or other tasks, we need to aggre the result at last
        
        self.call_hook('after_run')

    def resume(self, checkpoint, resume_optimizer = True):
        assert 'epoch_' in checkpoint
        res_dict = self.load_checkpoint(checkpoint, load_optim=resume_optimizer)

        self._epoch = res_dict['nums']  # 恢复epoch
        self.logger.info("resumed from epoch: {}".format(self._epoch))

        # 加载optim的state
        if resume_optimizer:
            self.logger.info("load optimizer's state dict")
            for submodule_name in self.optimizers_cfg.keys():
                self.model.optimizers[submodule_name].load_state_dict(res_dict[submodule_name])

    def save_checkpoint(self, out_dir, create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        filename_tmpl = "epoch_{}"
        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        self.logger.info('save checkpoint to {}'.format(filepath))
        mkdir_or_exist(filepath)
        if isinstance(self.model.optimizers, dict):
            for key in self.model.optimizers.keys():
                submodule = getattr(self.model, key, None)
                assert submodule is not None, "model should have submodule {}".format(key)
                assert isinstance(submodule, M.Module), "submodule should be instance of megengine.module.Module"
                mge.save(submodule.state_dict(), osp.join(filepath, key + module_ckpt_suffix))
                mge.save(self.model.optimizers[key].state_dict(), osp.join(filepath, key + optim_ckpt_suffix))
        else:
            raise TypeError(" the type of optimizers should be dict for save_checkpoint")
            
        if create_symlink:
            pass

    def register_training_hooks(self,
                                checkpoint_config,
                                log_config):
        """Register default hooks for epoch-based training.

        Default hooks include:

        - LrUpdaterHook
        - CheckpointSaverHook
        - logHook
        """
        if checkpoint_config is not None:
            checkpoint_config.setdefault('by_epoch', True)
        if log_config is not None:
            log_config.setdefault('by_epoch', False)

        self.register_checkpoint_hook(checkpoint_config)
        self.register_logger_hooks(log_config)

@BACKBONES.register_module()
class CenterTrack(M.Module):
    def __init__(self,
                 inp_h = 480,
                 inp_w = 480,
                 stride = 2,
                 channels = 32,
                 head_channels = 64,
                 backbone_type = "PoseResNet",
                 num_layers = 18,
                 num_classes = 1,
                 backbone_imagenet_pretrain = False,
                 all_pretrain = False,
                 all_pretrain_path = None,
                 min_overlap = 0.3,
                 fp = 0.1,
                 fn = 0.4):
        super(CenterTrack, self).__init__()
        assert backbone_type in backbone_spec.keys()
        if backbone_type == 'PoseResNet':
            self.backbone_out_c = 256
        else:
            self.backbone_out_c = 64

        support_strides = [2]
        assert stride in support_strides
        assert inp_h % stride == 0 and inp_w % stride == 0

        self.inp_h = inp_h
        self.inp_w = inp_w
        self.stride = stride
        self.backbone_type = backbone_type
        self.channels = channels
        self.head_channels = head_channels
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.backbone_imagenet_pretrain = backbone_imagenet_pretrain
        self.all_pretrain = all_pretrain
        self.all_pretrain_path = all_pretrain_path
        self.min_overlap = min_overlap
        self.fp = fp
        self.fn = fn
        self.hm_disturb = 0.05 # 用于对pre_hm进行扰动的比例

        self.fm_ctr = self.get_fm_ctr(inp_h, inp_w, stride)

        self.base_layer = M.Sequential(
                            M.Conv2d(3, channels, kernel_size=7, stride=1, padding=3, bias=False),
                            M.BatchNorm2d(channels),
                            M.ReLU()
                        ) # 480 -> 240
        self.pre_layer = M.Sequential(
                            M.Conv2d(3, channels, kernel_size=7, stride=1, padding=3, bias=False),
                            M.BatchNorm2d(channels),
                            M.ReLU()
                        )
        self.hm_layer = M.Sequential(
                            M.Conv2d(1, channels, kernel_size=7, stride=1, padding=3, bias=False),
                            M.BatchNorm2d(channels),
                            M.ReLU()
                        )

        self.init_backbone()
        self.init_heads()
        self.center_loss = Center_loss(alpha = 2, beta = 4)
        self.init_weights(pretrained = self.all_pretrain)

    def init_backbone(self):
        select_backbone = backbone_spec[self.backbone_type]
        self.backbone = select_backbone(num_layers = self.num_layers, inp = self.channels, pretrained = self.backbone_imagenet_pretrain)

    def init_heads(self):
        """
            init 3 kinds of heads: heat map, HW and motion
        """
        self.head_heatmap = M.Sequential(
            M.Conv2d(self.backbone_out_c, self.head_channels, 3, 1, 1, bias=True),
            M.ReLU(),
            M.Conv2d(self.head_channels, self.num_classes, 3, 1, 1, bias=True)
        )
        self.head_hw = M.Sequential(
            M.Conv2d(self.backbone_out_c, self.head_channels, 3, 1, 1, bias=True),
            M.ReLU(),
            M.Conv2d(self.head_channels, 2, 3, 1, 1, bias=True)
        )
        self.head_motion = M.Sequential(
            M.Conv2d(self.backbone_out_c, self.head_channels, 3, 1, 1, bias=True),
            M.ReLU(),
            M.Conv2d(self.head_channels, 2, 3, 1, 1, bias=True)
        )

    def forward(self, x, pre_img = None, pre_hm = None):
        x = self.base_layer(x)
        if pre_img is not None:
            pre_img = self.pre_layer(pre_img)
            x = x + pre_img
        if pre_hm is not None:
            pre_hm = self.hm_layer(pre_hm)
            x = x + pre_hm

        x = self.backbone(x)

        heatmap = self.head_heatmap(x)
        hw = self.head_hw(x)
        motion = self.head_motion(x)
        heatmap = F.sigmoid(heatmap)
        return heatmap, hw, motion

    def get_fm_ctr(self, inph, inpw, stride):
        fm_height, fm_width = inph // stride, inpw // stride # 240, 240
        y_list = np.linspace(0., fm_height - 1., fm_height).reshape(1, 1, fm_height, 1)
        y_list = y_list.repeat(fm_width, axis=3)
        x_list = np.linspace(0., fm_width - 1., fm_width).reshape(1, 1, 1, fm_width)
        x_list = x_list.repeat(fm_height, axis=2)
        xy_list = (self.stride - 1.0)/2 + np.concatenate((x_list, y_list), 1) * self.stride
        xy_list = xy_list.astype(np.float32)
        return xy_list

    def loss_hw(self, pred_hw, gt_hw, gt_mask):
        """
            pred_hw: b,2,h,w
            gt_hw: b,2,h,w
            gt_mask: b,1,h,w
            注意要batchwise的做，因为每个sample的框数量不一样
        """
        bs, _, _, _ = pred_hw.shape
        loss = gt_mask * F.abs(pred_hw - gt_hw) # [b,2,h,w]
        loss = loss.reshape(bs, -1)
        loss = F.sum(loss, axis=1, keepdims=True) / 2 # [b, 1]
        gt_mask = gt_mask.reshape(bs, -1)
        gt_mask = F.sum(gt_mask, axis=1, keepdims=True) # [b, 1]
        loss = loss / gt_mask
        loss = loss.sum() / bs
        return loss

    def loss_motion(self, pred_motion, gt_motion, gt_mask):
        """
            对于给定的前一帧和当前帧，根据同一id的位置变化，来学习motion
        """
        bs, _, _, _ = pred_motion.shape
        loss = gt_mask * F.abs(pred_motion - gt_motion) # [b,2,h,w]
        loss = loss.reshape(bs, -1)
        loss = F.sum(loss, axis=1, keepdims=True) / 2 # [b, 1]
        gt_mask = gt_mask.reshape(bs, -1)
        gt_mask = F.sum(gt_mask, axis=1, keepdims=True) # [b, 1]
        loss = loss / gt_mask
        loss = loss.sum() / bs
        return loss

    def get_loss(self, pred_heatmap, pred_hw, pred_motion, gt_bboxes, gt_labels, loss_weight, pre_gt_bboxes=None, pre_gt_labels = None):
        """
            given pre hm and now bbox, cal loss
        """
        gt_hms, gt_hw, gt_mask, gt_motion = self.get_targets(gt_bboxes, gt_labels, pre_gt_bboxes, pre_gt_labels)
        loss_hms = self.center_loss(pred_heatmap, gt_hms)
        loss_hw = self.loss_hw(pred_hw, gt_hw, gt_mask)
        loss_motion = self.loss_motion(pred_motion, gt_motion, gt_mask)
        weight_hms = loss_weight['hms']
        weight_hw = loss_weight['hw']
        weight_motion = loss_weight['motion']
        total_loss = weight_hms * loss_hms + weight_hw * loss_hw + weight_motion * loss_motion
        return [loss_hms, loss_hw, loss_motion, total_loss]

    def get_gaussian_radius(self, det_size, min_overlap):
        """
            for centernet, only have one case
            refer to https://github.com/princeton-vl/CornerNet/blob/e5c39a31a8abef5841976c8eab18da86d6ee5f9a/sample/utils.py
        """
        assert min_overlap > 0.05 and min_overlap < 0.95
        height, width = det_size

        a1  = 1
        b1  = (height + width)
        c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1  = (b1 - sq1) / (2 * a1)
        return r1

    def map_to_feat(self, x_or_y, w_or_h, factor = 1):
        """
            加一定概率的扰动,根据 self.hm_disturb
        """
        x_or_y_int = x_or_y + np.random.randn() * self.hm_disturb * w_or_h * factor
        x_or_y_int = (x_or_y_int - (self.stride-1.0)/2) / self.stride
        x_or_y_int = int(x_or_y_int + 0.5)
        return x_or_y_int

    def get_test_pre_hm(self, pre_gt_bboxes):
        bs = len(pre_gt_bboxes)
        assert bs == 1
        _, _, feat_h, feat_w = self.fm_ctr.shape
        pre_hms = F.zeros((bs, self.num_classes, feat_h, feat_w))
        for batch_id in range(bs):
            gt_bbox = pre_gt_bboxes[batch_id] # [S, 4]
            # if len(gt_bbox) > 5:
            #     gt_bbox = gt_bbox[0: len(gt_bbox)-5, :]
            center_x = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2 # w
            center_y = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2 # h
            for gt_id in range(center_x.shape[0]):
                # get radius
                box_w = gt_bbox[gt_id][2] - gt_bbox[gt_id][0]
                box_h = gt_bbox[gt_id][3] - gt_bbox[gt_id][1]
                scale_box_w = box_w / self.stride
                scale_box_h = box_h / self.stride
                # 在给定hw的情况下，决定一个radius，使得iou至少为min_overlap
                radius = self.get_gaussian_radius([scale_box_h, scale_box_w], self.min_overlap)
                radius = max(0, int(radius))
                ind = 0

                ctx = center_x[gt_id]
                cty = center_y[gt_id]
                ctx1_int = self.map_to_feat(ctx, box_w, factor=0)
                cty1_int = self.map_to_feat(cty, box_h, factor=0)
                pre_hms[batch_id, ind] = gen_gaussian_target(pre_hms[batch_id, ind], [ctx1_int, cty1_int], radius)

        pre_hms = F.vision.interpolate(pre_hms, scale_factor=self.stride, align_corners=False)
        return pre_hms

    def get_pre_hm(self, pre_gt_bboxes, pre_gt_labels):
        """
            根据pre_gt_bboxes，生成pre hm，这里使用feat进行bilinear上采样到输入图片的尺寸
            pre_gt_bboxes: list of ndarray [[S, 4] ....]  the 4 is [tl_x, tl_y, br_x, br_y]  float64
            pre_gt_labels: list of ndarray  [[S, 2] ....]  the 2 is [class, id]               int64

            # 对于gt_bboxes1有三种增强方法 按照下面的顺序执行
            1.一定概率消失, 什么也不加       self.fn
            2.按照正太分布随机移动框的位置（移动后不超边界）
            3.一定概率在自己的周围再加一个框  self.fp
        """
        bs = len(pre_gt_bboxes)
        _, _, feat_h, feat_w = self.fm_ctr.shape
        pre_hms = F.zeros((bs, self.num_classes, feat_h, feat_w))
        for batch_id in range(bs):
            gt_bbox = pre_gt_bboxes[batch_id] # [S, 4]
            gt_label = pre_gt_labels[batch_id] # [S, 2]
            center_x = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2 # w
            center_y = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2 # h
            for gt_id in range(center_x.shape[0]):
                if np.random.random() < self.fn:
                    continue
                # get radius
                box_w = gt_bbox[gt_id][2] - gt_bbox[gt_id][0]
                box_h = gt_bbox[gt_id][3] - gt_bbox[gt_id][1]
                scale_box_w = box_w / self.stride
                scale_box_h = box_h / self.stride
                # 在给定hw的情况下，决定一个radius，使得iou至少为min_overlap
                radius = self.get_gaussian_radius([scale_box_h, scale_box_w], self.min_overlap)
                radius = max(0, int(radius))
                ind = gt_label[gt_id][0]

                ctx = center_x[gt_id]
                cty = center_y[gt_id]
                ctx1_int = self.map_to_feat(ctx, box_w)
                cty1_int = self.map_to_feat(cty, box_h)
                pre_hms[batch_id, ind] = gen_gaussian_target(pre_hms[batch_id, ind], [ctx1_int, cty1_int], radius)
                # 一定概率再生成一个
                if np.random.random() < self.fp:
                    ctx2_int = self.map_to_feat(ctx, box_w, factor=2)
                    cty2_int = self.map_to_feat(cty, box_h, factor=2)
                    pre_hms[batch_id, ind] = gen_gaussian_target(pre_hms[batch_id, ind], [ctx2_int, cty2_int], radius)

        pre_hms = F.vision.interpolate(pre_hms, scale_factor=self.stride, align_corners=False)
        return pre_hms

    def get_targets(self, gt_bboxes, gt_labels, pre_gt_bboxes = None, pre_gt_labels = None):
        """
            gt_bboxes: list of ndarray  [[S, 4] ....]  the 4 is [tl_x, tl_y, br_x, br_y]  float64
            gt_labels: list of ndarray  [[S, 2] ....]  the 2 is [class, id]               int64
 
            return: (gt_hms, gt_hw, gt_mask)
            gt_hms: [B, classes, h, w]
            gt_hw: [B, 2, h, w]
            gt_mask: [B, 1, h, w]
        """
        assert len(gt_bboxes) == len(gt_labels)
        bs = len(gt_bboxes)
        _, _, feat_h, feat_w = self.fm_ctr.shape

        gt_hms = F.zeros((bs, self.num_classes, feat_h, feat_w))
        gt_hw = F.zeros((bs, 2, feat_h, feat_w))
        gt_mask = F.zeros((bs, 1, feat_h, feat_w))
        gt_motion = None
        
        if pre_gt_bboxes is not None:
            assert pre_gt_labels is not None
            assert len(pre_gt_labels) == len(pre_gt_bboxes) and len(pre_gt_bboxes) == len(gt_bboxes)
            gt_motion = F.zeros((bs, 2, feat_h, feat_w))

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id] # [S, 4]
            gt_label = gt_labels[batch_id] # [S, 2]
            # 算出每一个bbox的中心位置
            center_x = (gt_bbox[:, 0] + gt_bbox[:, 2]) / 2 # w
            center_y = (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2 # h

            if pre_gt_bboxes is not None:
                # 构造一个字典根据id可以查到上一帧有没有以及有的话，中心位置是多少(小数坐标)
                id_map = {}
                pre_gt_bbox = pre_gt_bboxes[batch_id]
                pre_gt_label = pre_gt_labels[batch_id]
                pre_center_x = (pre_gt_bbox[:, 0] + pre_gt_bbox[:, 2]) / 2 # w
                pre_center_y = (pre_gt_bbox[:, 1] + pre_gt_bbox[:, 3]) / 2 # h
                for gt_id in range(pre_center_x.shape[0]):
                    ID = pre_gt_label[gt_id][1] # ID
                    ctx = pre_center_x[gt_id]
                    ctx = (ctx - (self.stride-1.0)/2) / self.stride
                    cty = pre_center_y[gt_id]
                    cty = (cty - (self.stride-1.0)/2) / self.stride
                    id_map[ID] = (ctx, cty) # 中心在feat中的位置

            for gt_id in range(center_x.shape[0]):
                # get ctx_int, cty_int, 根据gt_centers，找到最接近的在feat中的坐标（可能会出现距离多个点一样的情况）
                ctx = center_x[gt_id]
                ctx = (ctx - (self.stride-1.0)/2) / self.stride
                ctx_int = int(ctx + 0.5)
                cty = center_y[gt_id]
                cty = (cty - (self.stride-1.0)/2) / self.stride
                cty_int = int(cty + 0.5)
                # get radius
                scale_box_w = (gt_bbox[gt_id][2] - gt_bbox[gt_id][0]) / self.stride
                scale_box_h = (gt_bbox[gt_id][3] - gt_bbox[gt_id][1]) / self.stride
                # 在给定hw的情况下，决定一个radius，使得iou至少为min_overlap
                radius = self.get_gaussian_radius([scale_box_h, scale_box_w], self.min_overlap)
                radius = max(0, int(radius))
                ind = gt_label[gt_id][0]
                gt_hms[batch_id, ind] = gen_gaussian_target(gt_hms[batch_id, ind], [ctx_int, cty_int], radius)

                gt_hw[batch_id, 0, cty_int, ctx_int] = scale_box_w
                gt_hw[batch_id, 1, cty_int, ctx_int] = scale_box_h

                gt_mask[batch_id, 0, cty_int, ctx_int] = 1.0                

                # 如果当前id在上一帧中也有，则看其位置在哪里，构造gt_motion的值，如果没有则置motion为0（或者100？）
                if gt_motion is not None:
                    # 拿当前id去字典中找
                    ID = gt_label[gt_id][1]
                    if ID in id_map.keys():
                        # 前一个位置减后一个位置
                        motion = id_map[ID]
                        gt_motion[batch_id, 0, cty_int, ctx_int] = motion[0] - ctx
                        gt_motion[batch_id, 1, cty_int, ctx_int] = motion[1] - cty
                    else:
                        pass # 默认就是0

        return gt_hms, gt_hw, gt_mask, gt_motion

    def init_weights(self, pretrained):
        if pretrained:
            assert self.all_pretrain_path is not None
            assert ".mge" in self.all_pretrain_path
            print("loading pretrained model for all module 🤡🤡🤡🤡🤡🤡...")
            state_dict = megengine.load(self.all_pretrain_path)
            self.load_state_dict(state_dict, strict=True)
        else:
            default_init_weights(self.base_layer)
            default_init_weights(self.pre_layer)
            default_init_weights(self.hm_layer)
            default_init_weights(self.head_heatmap)
            default_init_weights(self.head_hw)
            default_init_weights(self.head_motion)
            
            def set_bias(m):
                assert isinstance(m, M.Conv2d)
                M.init.fill_(m.bias, -2.19)

            set_bias(self.head_heatmap[-1])

class BaseModel(M.Module):
    """Base model.

    All models should subclass it.
    All subclass should overwrite:

        ``init_weights``, supporting to initialize models.

        ``train_step``, supporting to train one step when training.

        ``test_step``, supporting to test(predict) one step for eval and test.

        ``cal_for_eval``, do some calculation for eval.
    """

    def __init__(self):
        super(BaseModel, self).__init__()
        self.local_rank = get_rank()

    def forward(self, *inputs, **kwargs):
        pass
    
    @abstractmethod
    def init_weights(self):
        """Abstract method for initializing weight.

        All subclass should overwrite it.
        """
        pass

    @abstractmethod
    def train_step(self, batchdata):
        """Abstract method for one training step.

        All subclass should overwrite it.
        """
        pass

    @abstractmethod
    def test_step(self, batchdata, **kwargs):
        """Abstract method for one test step.

        All subclass should overwrite it.
        """
        pass

    @abstractmethod
    def cal_for_eval(self, gathered_outputs, gathered_batchdata):
        pass

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

        self.train_generator_batch = None
        self.test_generator_batch = test_batch

        # get workdir 
        workdir = kwargs.get("workdir", "./workdir")
        self.summary = {}
        self.gen_writer = None

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

@PIPELINES.register_module()
class Compose(object):
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f'transform must be callable or a dict, '
                                f'but got {type(transform)}')

    def __call__(self, data):
        """Call function.

        Args:
            data (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A tuple (image, label) containing the processed data and label.
        """
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string

@DATASETS.register_module()
class RepeatDataset(Dataset):
    """A wrapper of MapDataset dataset to repeat.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        super(RepeatDataset, self).__init__()
        self.dataset = dataset
        self.times = times
        dataset.logger.info("use repeatdataset, repeat times: {}".format(times))
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """Length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return self.times * self._ori_len

import random
from functools import partial
import numpy as np

def build_dataset(cfg, default_args=None):
    """Build a dataset from config dict.

    It supports a variety of dataset config. If ``cfg`` is a Sequential (list
    or dict), it will be a concatenated dataset of the datasets specified by
    the Sequential. If it is a ``RepeatDataset``, then it will repeat the
    dataset ``cfg['dataset']`` for ``cfg['times']`` times. If the ``ann_file``
    of the dataset is a Sequential, then it will build a concatenated dataset
    with the same dataset type but different ``ann_file``.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict, optional): Default initialization arguments.
            Default: None.

    Returns:
        Dataset: The constructed dataset.
    """
    if isinstance(cfg, (list, tuple)):
        raise NotImplementedError("dose not support list(tuple) configs for dataset build now")
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        raise NotImplementedError("does not support list(tuple) ann_files for dataset build now")
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset

def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(f'the cfg dict must contain the key "type", but got {cfg}')
    if not isinstance(registry, Registry):
        raise TypeError('registry must be an edit.utils.Registry object, '
                        f'but got {type(registry)}')
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError('default_args must be a dict or None, '
                        f'but got {type(default_args)}')

    args = cfg.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f'type must be a str or valid type, but got {type(obj_type)}')

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_cls(**args)

def build(cfg, registry, default_args=None):
    """Build module function.

    Args:
        cfg (dict): Configuration for building modules.
        registry (obj): ``registry`` object.
        default_args (dict, optional): Default arguments. Defaults to None.
    """
    if isinstance(cfg, list):
        raise NotImplementedError("list of cfg does not support now")
        # modules = [
        #     build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        # ]
        # return Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)

def build_backbone(cfg):
    """Build backbone.

    Args:
        cfg (dict): Configuration for building backbone.
    """
    return build(cfg, BACKBONES)

def build_model(cfg, workdir, train_cfg=None, eval_cfg=None):
    """Build model.

    Args:
        cfg (dict): Configuration for building model.
        train_cfg (dict): Training configuration. Default: None.
        eval_cfg (dict): Testing configuration. Default: None.
    """
    return build(cfg, MODELS, dict(train_cfg=train_cfg, eval_cfg=eval_cfg, workdir = workdir))

logger_initialized = {}

def get_logger(name, log_file=None, log_level=logging.INFO):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    
    if name in logger_initialized:
        return logger

    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name): # child
            return logger

    # fix stream twice bug
    # while logger.handlers:
    #     logger.handlers.pop()

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if is_distributed():
        rank = get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        handlers.append(file_handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True
    return logger

def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. 
    By default a StreamHandler will be added. 
    If `log_file` is specified, a FileHandler will also be added. 
    The name of the root logger is the top-level package name, e.g., "edit".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    root_name = __name__.split('.')[0]  # edit.utils.logger
    if is_distributed():
        rank = get_rank()
        root_name = "rank" + str(rank) + "_" + root_name
    logger = get_logger(root_name, log_file, log_level)
    return logger

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Test an editor o(*￣▽￣*)ブ')
    parser.add_argument('--result_save_path', type=str, default=None, help='the dir to save logs and models')
    parser.add_argument('--test_dataset_path', type=str, default=None, help='')
    parser.add_argument('--weights_path', type=str, default=None, help='')
    args = parser.parse_args()
    return args

def get_loader(dataset):
    samples_per_gpu = 1
    workers_per_gpu = 5
    sampler = SequentialSampler(dataset, batch_size=samples_per_gpu, drop_last=False)
    loader = DataLoader(dataset, sampler, num_workers=workers_per_gpu)
    return loader

def test(model, datasets, work_dir):
    data_loaders = [ get_loader(ds) for ds in datasets ]
    runner = EpochBasedRunner(model=model, optimizers_cfg=dict(generator=dict(type='Adam', lr=1e-4, betas=(0.99, 0.99))), work_dir=work_dir)
    runner.load_checkpoint(load_from, load_optim=False)
    runner.sync_model_params()
    runner.run(data_loaders, 'test', 1)

def worker(rank, world_size, work_dir):
    trace.enabled = False
    cfg_model = dict(
        type='ONLINE_MOT',
        generator=dict(
            type='CenterTrack',
            inp_h = input_h,
            inp_w = input_w,
            channels = 32,
            head_channels = 64,
            backbone_type = "DLA_GN",
            num_classes = 1,
            backbone_imagenet_pretrain = False,
            all_pretrain = False,
            min_overlap = 0.3,
            fp = fp,
            fn = fn),
        loss_weight = loss_weight
    )
    cfg_eval_cfg = dict(metrics=['MOTA', 'IDF1'])
    img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])
    test_dataset_type = 'MotFishTestDataset'
    test_pipeline = [
        dict(
            type='LoadImageFromFile',
            io_backend='disk',
            key='img',
            flag='color'),
        dict(type='Add_contrast', keys=['img'], value = 0.9),
        dict(type='RescaleToZeroOne', keys=['img']),
        dict(type='Resize', keys=['img'], size=[input_h, input_w], interpolation="area"),
        dict(type='Normalize', keys=['img'], to_rgb=True, **img_norm_cfg),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img', 'scale_factor', 'index', 'clipname', 'gap', 'total_len'])
    ]
    cfg_data_test = dict(
        type=test_dataset_type,
        folder= dataroot,
        pipeline=test_pipeline)
    model = build_model(cfg_model, work_dir, eval_cfg=cfg_eval_cfg)  # eval cfg can provide some useful info, e.g. the padding multi
    datasets = [build_dataset(cfg_data_test)]
    test(model, datasets, work_dir)

def main():
    args = parse_args()
    if args.result_save_path is not None:
        PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
        work_dir = os.path.join(PROJECT_PATH, args.result_save_path)
    else:
        raise RuntimeError("")

    global load_from
    load_from = args.weights_path
    global dataroot
    dataroot = args.test_dataset_path
    dataroot = os.path.abspath(dataroot)
    print(dataroot)
    # dataroot = os.path.join(dataroot, 'preliminary', 'test')
    mkdir_or_exist(os.path.abspath(work_dir))
    log_file = os.path.join(work_dir, 'root.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')
    mge.set_default_device(device='gpu0')
    worker(0, 1, work_dir)
    os.remove(os.path.join(work_dir, 'root.log'))

if __name__ == "__main__":
    main()

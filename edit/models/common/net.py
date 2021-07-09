import numpy as np
import megengine
import megengine.module as M
import megengine.functional as F
from edit.models.common import ShuffleV2Block, CoordAtt
import math
from . import default_init_weights

# class GCT(M.Module):
#     def __init__(self, num_channels, epsilon=1e-5):
#         super(GCT, self).__init__()
#         self.alpha = megengine.Parameter(np.ones((1, num_channels, 1, 1), dtype=np.float32))
#         self.gamma = megengine.Parameter(np.zeros((1, num_channels, 1, 1), dtype=np.float32))
#         self.beta = megengine.Parameter(np.zeros((1, num_channels, 1, 1), dtype=np.float32))
#         self.epsilon = epsilon

#     def forward(self, x):
#         embedding = ((F.sum((x**2), axis=[2,3], keepdims=True) + self.epsilon)**(0.5)) * self.alpha
#         norm = self.gamma / ((F.mean((embedding**2), axis=1, keepdims=True) + self.epsilon)**(0.5))
#         gate = 1. + F.tanh(embedding * norm + self.beta)
#         return x * gate

# class SEL(M.Module):
#     def __init__(self, hidden):
#         super(SEL, self).__init__()
#         self.conv = M.Conv2d(hidden, hidden, 1, 1)
#         self.relu = M.ReLU()
#         self.init_weights()

#     def forward(self, x):
#         return x * F.sigmoid( self.conv(self.relu(x)) )

#     def init_weights(self):
#         default_init_weights(self.conv, scale=0.1)

class MobileNeXt(M.Module):
    def __init__(self, in_channels, out_channels, stride = 1, reduction = 4):
        """
            默认使用coordinate attention在第一个dwise之后
            https://github.com/Andrew-Qibin/CoordAttention/blob/main/coordatt.py
        """
        super(MobileNeXt, self).__init__()
        self.stride = stride
        assert stride in (1, 2)
        assert in_channels % reduction == 0
        mid_channel = in_channels // reduction

        self.dconv1 = M.Sequential(
            M.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),
            # M.InstanceNorm(in_channels),
            M.ReLU()
        )
        self.CA = CoordAtt(inp = in_channels, oup = in_channels)
        self.conv1 = M.Conv2d(in_channels, mid_channel, kernel_size=1, stride=1, padding=0)
        self.conv2 = M.Sequential(
            M.Conv2d(mid_channel, out_channels, kernel_size=1, stride=1, padding=0),
            M.InstanceNorm(out_channels),
            M.ReLU()
        )
        self.dconv2 = M.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=out_channels)
        if stride > 1:
            self.downsample = M.Sequential(
                M.Conv2d(in_channels, out_channels, kernel_size=1, stride = stride, padding=0)
            )
        self.init_weights()

    def init_weights(self):
        for m in [self.conv2, self.dconv1]:
            default_init_weights(m, scale=0.1)

    def forward(self, x):
        identity = x
        if self.stride > 1:
            identity = self.downsample(identity)
        x = self.CA(self.dconv1(x))
        out = self.dconv2(self.conv2(self.conv1(x)))
        return identity + out

class ResBlock(M.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, init_scale = 0.1):
        super(ResBlock, self).__init__()
        self.init_scale = init_scale
        self.conv1 = M.ConvRelu2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size//2))
        self.conv2 = M.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size//2))
        self.init_weights()

    def init_weights(self):
        for m in [self.conv1, self.conv2]:
            default_init_weights(m, scale=self.init_scale)

    def forward(self, x):
        identity = x
        out = self.conv2(self.conv1(x))
        return identity + out

class RK4_ResBlock(M.Module):
    """
        1block = 8 conv = 4 resblocks
        https://arxiv.org/pdf/2103.15244.pdf
    """
    def __init__(self, chs, kernel_size=3, init_scale = 0.1):
        super(RK4_ResBlock, self).__init__()
        self.init_scale = init_scale
        self.block1 = M.Sequential(
            M.ConvRelu2d(chs, chs, kernel_size=kernel_size, stride=1, padding=(kernel_size//2)),
            M.Conv2d(chs, chs, kernel_size=kernel_size, stride=1, padding=(kernel_size//2))
        )
        self.block2 = M.Sequential(
            M.ConvRelu2d(chs, chs, kernel_size=kernel_size, stride=1, padding=(kernel_size//2)),
            M.Conv2d(chs, chs, kernel_size=kernel_size, stride=1, padding=(kernel_size//2))
        )
        self.block3 = M.Sequential(
            M.ConvRelu2d(chs, chs, kernel_size=kernel_size, stride=1, padding=(kernel_size//2)),
            M.Conv2d(chs, chs, kernel_size=kernel_size, stride=1, padding=(kernel_size//2))
        )
        self.block4 = M.Sequential(
            M.ConvRelu2d(chs, chs, kernel_size=kernel_size, stride=1, padding=(kernel_size//2)),
            M.Conv2d(chs, chs, kernel_size=kernel_size, stride=1, padding=(kernel_size//2))
        )
        self.init_weights()

    def init_weights(self):
        for m in [self.block1, self.block2, self.block3, self.block4]:
            default_init_weights(m, scale=self.init_scale)

    def forward(self, x):
        out1 = self.block1(x) + x
        out2 = self.block2(0.5 * out1) + x
        out3 = self.block3(0.5 * out2) + x
        out4 = self.block4(out3) + x
        return x + (out1 + 2*out2 + 2*out3 + out4) / 6

class ResBlocks(M.Module):
    def __init__(self, channel_num, resblock_num, kernel_size=3, blocktype="resblock"):
        super(ResBlocks, self).__init__()
        assert blocktype in ("resblock", "shuffleblock", "MobileNeXt", "RK4")
        if blocktype == "resblock":
            self.model = M.Sequential(
                self.make_resblock_layer(channel_num, resblock_num, kernel_size),
            )
        elif blocktype == "shuffleblock":
            self.model = M.Sequential(
                self.make_shuffleblock_layer(channel_num, resblock_num, kernel_size),
            )
        elif blocktype == "MobileNeXt":
            self.model = M.Sequential(
                self.make_MobileNeXt_layer(channel_num, resblock_num, kernel_size)
            )
        elif blocktype == "RK4":
            assert resblock_num % 4 == 0, "you should make sure resblock_num is multiple of 4"
            self.model = M.Sequential(
                self.make_RK4_layer(channel_num, resblock_num // 4, kernel_size)
            )
        else:
            raise NotImplementedError("")

    def make_MobileNeXt_layer(self, ch_out, num_blocks, kernel_size):
        layers = []
        for _ in range(num_blocks):
            layers.append(MobileNeXt(ch_out, ch_out, kernel_size))
        return M.Sequential(*layers)

    def make_resblock_layer(self, ch_out, num_blocks, kernel_size):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResBlock(ch_out, ch_out, kernel_size))
        return M.Sequential(*layers)

    def make_shuffleblock_layer(self, ch_out, num_blocks, kernel_size):
        layers = []
        for _ in range(num_blocks):
            layers.append(ShuffleV2Block(inp = ch_out//2, oup=ch_out, mid_channels=ch_out//2, ksize=kernel_size, stride=1))
        return M.Sequential(*layers)

    def make_RK4_layer(self, ch_out, num_blocks, kernel_size):
        layers = []
        for _ in range(num_blocks):
            layers.append(RK4_ResBlock(chs = ch_out, kernel_size=kernel_size, init_scale=0.1))
        return M.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


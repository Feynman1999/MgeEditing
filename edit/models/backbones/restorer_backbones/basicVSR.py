import numpy as np
import megengine
import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
from edit.models.builder import BACKBONES
import math

backwarp_tenGrid = {}

def pad_W(x):
    # 在图像右方添一列 replicate模式
    # x: [B,C,H,W]
    return F.concat([x, x[:,:,:,-1:]], axis=3)

def pad_H(x):
    # 在图像下方添一行 replicate模式
    return F.concat([x, x[:,:,-1:, :]], axis=2)


def backwarp(tenInput, tenFlow):
    _, _, H, W = tenFlow.shape
    if str(tenFlow.shape) not in backwarp_tenGrid.keys():
        x_list = np.linspace(0., W - 1., W).reshape(1, 1, 1, W)
        x_list = x_list.repeat(H, axis=2)
        y_list = np.linspace(0., H - 1., H).reshape(1, 1, H, 1)
        y_list = y_list.repeat(W, axis=3)
        xy_list = np.concatenate((x_list, y_list), 1)  # [1,2,H,W]
        backwarp_tenGrid[str(tenFlow.shape)] = megengine.tensor(xy_list.astype(np.float32))
    return F.nn.remap(inp = tenInput, map_xy=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).transpose(0, 2, 3, 1), border_mode="REPLICATE")

class Basic(M.Module):
    def __init__(self, intLevel):
        super(Basic, self).__init__()

        self.netBasic = M.Sequential(
            Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), # 8=3+3+2
            M.ReLU(),
            Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            M.ReLU(),
            Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
            M.ReLU(),
            Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
            M.ReLU(),
            Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
        )

    def forward(self, tenInput):
        return self.netBasic(tenInput)

class Spynet(M.Module):
    def __init__(self, num_layers, pretrain_ckpt_path = None):
        super(Spynet, self).__init__()
        assert num_layers in (6, )
        self.pretrain_ckpt_path = pretrain_ckpt_path
        basic_list = [ Basic(intLevel) for intLevel in range(num_layers) ] # 在本次VSR任务中，最多只会用到0~4
        self.netBasic = M.Sequential(*basic_list)

    def preprocess(self, tenInput):
        tenRed = (tenInput[:, 0:1, :, :] - 0.485) / 0.229
        tenGreen = (tenInput[:, 1:2, :, :] - 0.456) / 0.224
        tenBlue = (tenInput[:, 2:3, :, :] - 0.406 ) / 0.225
        return F.concat([tenRed, tenGreen, tenBlue], axis=1) # [B,3,H,W]

    def forward(self, tenFirst, tenSecond):
        tenFirst = [self.preprocess(tenFirst)]
        tenSecond = [self.preprocess(tenSecond)]

        # 构造图像金字塔,最多加5个，也就是一共6个
        for intLevel in range(5):
            if tenFirst[0].shape[2] > 4 or tenFirst[0].shape[3] > 4:
                tenFirst.insert(0, F.avg_pool2d(inp=tenFirst[0], kernel_size=2, stride=2))
                tenSecond.insert(0, F.avg_pool2d(inp=tenSecond[0], kernel_size=2, stride=2))
        
        tenFlow = F.zeros([tenFirst[0].shape[0], 2, int(math.floor(tenFirst[0].shape[2] / 2.0)), int(math.floor(tenFirst[0].shape[3] / 2.0))])
        # print(len(tenFirst))
        for intLevel in range(len(tenFirst)): # 5 for training (4*4, 8*8, 16*16, 32*32, 64*64)       6 for test  (6*10, 12*20, 24*40, 48*80, 96*160, 192*320)
            tenUpsampled = F.nn.interpolate(inp=tenFlow, scale_factor=2, mode='BILINEAR', align_corners=True) * 2.0
            if tenUpsampled.shape[2] != tenFirst[intLevel].shape[2]:
                tenUpsampled = pad_H(tenUpsampled)
            if tenUpsampled.shape[3] != tenFirst[intLevel].shape[3]:
                tenUpsampled = pad_W(tenUpsampled)
            tenFlow = self.netBasic[intLevel]( F.concat([tenFirst[intLevel], backwarp(tenInput=tenSecond[intLevel], tenFlow=tenUpsampled), tenUpsampled], axis=1) ) + tenUpsampled
        return tenFlow

    def init_weights(self, strict=True):
        # load ckpt from path
        if self.pretrain_ckpt_path is not None:
            print("loading pretrained model for Spynet...")
            state_dict = megengine.load(self.pretrain_ckpt_path)
            self.load_state_dict(state_dict, strict=strict)


class ResBlock(M.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='lrelu'):
        super(ResBlock, self).__init__()

        if activation == 'relu':
            self.act = M.ReLU()
        elif activation == 'prelu':
            self.act = M.PReLU(num_parameters=1, init=0.25)
        elif activation == 'lrelu':
            self.act = M.LeakyReLU(negative_slope=0.2)
        else:
            raise NotImplementedError("not implemented activation")

        m = []
        m.append(M.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size//2)))
        m.append(self.act)
        m.append(M.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size//2)))
        self.body = M.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x
        return self.act(res)


class ResBlocks(M.Module):
    def __init__(self, channel_num, resblock_num, kernel_size=3, activation='lrelu'):
        super(ResBlocks, self).__init__()
        self.model = M.Sequential(
            self.make_layer(ResBlock, channel_num, resblock_num, kernel_size, activation),
        )

    def make_layer(self, block, ch_out, num_blocks, kernel_size, activation):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(ch_out, ch_out, kernel_size, activation))
        return M.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PixelShuffle(M.Module):
    def __init__(self, scale=2):
        super(PixelShuffle, self).__init__()
        self.scale = scale

    def forward(self, inputs):
        # N C iH iW
        N, C, iH, iW = inputs.shape
        oH = iH * self.scale
        oW = iW * self.scale
        oC = C // (self.scale ** 2)
        # N C s s iH iW
        output = inputs.reshape(N, oC, self.scale, self.scale, iH, iW)
        # N C iH s iW s
        output = output.transpose(0, 1, 4, 3, 5, 2)
        # N C oH oW
        output = output.reshape(N, oC, oH, oW)
        return output


@BACKBONES.register_module()
class BasicVSR(M.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, blocknums, upscale_factor, pretrained_optical_flow_path):
        super(BasicVSR, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.blocknums = blocknums
        self.upscale_factor = upscale_factor

        self.flownet = Spynet(num_layers=6, pretrain_ckpt_path=pretrained_optical_flow_path)
        self.conv = Conv2d(in_channels=self.hidden_channels + self.in_channels, out_channels=self.hidden_channels, kernel_size=3, stride=1, padding=1)
        self.feature_extracter = ResBlocks(channel_num=hidden_channels, resblock_num=self.blocknums)
        self.upsampler =  M.Sequential(
            M.ConvRelu2d(2*self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            M.ConvRelu2d(self.hidden_channels, 256, kernel_size=3, stride=1, padding=1),
            M.ConvRelu2d(256, 512, kernel_size=3, stride=1, padding=1),
            PixelShuffle(scale=2),
            M.ConvRelu2d(128, 256, kernel_size=3, stride=1, padding=1),
            PixelShuffle(scale=2),
            M.ConvRelu2d(64, 64, kernel_size=3, stride=1, padding=1),
            M.Conv2d(64, self.out_channels, kernel_size=3, stride=1, padding=1)
        )

    def do_upsample(self, forward_hidden, backward_hidden):
        # 处理某一个time stamp的Hidden
        return self.upsampler(F.concat([forward_hidden, backward_hidden], axis=1))  # [B, 3, 4*H, 4*W]

    def forward(self, hidden, flow, now_frame):
        # hidden [B, C, H, W]
        mid_hidden = backwarp(hidden, flow) # [B, C, H, W]
        mid_hidden = self.conv(F.concat([now_frame, mid_hidden], axis=1))
        mid_hidden = self.feature_extracter(mid_hidden)
        return mid_hidden

    def init_weights(self, pretrained):
        self.flownet.init_weights()

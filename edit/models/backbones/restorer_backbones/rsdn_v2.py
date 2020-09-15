"""
for MOMM model
使用RNN + window + SD
"""

import numpy as np
import megengine
import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
from edit.models.builder import BACKBONES
from edit.models.common import add_H_W_Padding

class AU(M.Module):
    def __init__(self, ch):
        super(AU, self).__init__()
        self.conv = M.Sequential(
            M.Conv2d(2*ch, ch, kernel_size=5, stride=1, padding=2),
            M.LeakyReLU()
        )

    def forward(self, mid, ref):
        """
            mid: [B, C, H, W]
            ref: [B, C, H, W]
            return: [B, C, H, W]
        """
        mid = F.concat([mid, ref], axis = 1)
        return self.conv(mid)

class CARBBlock(M.Module):
    def __init__(self, channel_num):
        super(CARBBlock, self).__init__()
        self.conv1 = M.Sequential(
            M.Conv2d(channel_num, channel_num, kernel_size=3, padding=1, stride=1),
            M.LeakyReLU(),
            M.Conv2d(channel_num, channel_num, kernel_size=3, padding=1, stride=1),
        )
        # self.global_average_pooling = nn.AdaptiveAvgPool2d((1,1))  # B,C,H,W -> B,C,1,1
        self.linear = M.Sequential(
            M.Linear(channel_num, channel_num // 2),
            M.LeakyReLU(),
            M.Linear(channel_num // 2, channel_num),
            M.Sigmoid()
        )
        self.conv2 = M.Conv2d(channel_num*2, channel_num, kernel_size=1, padding=0, stride=1)
        self.lrelu = M.LeakyReLU()

    def forward(self, x):
        x1 = self.conv1(x)  # [B, C, H, W]
        
        w = F.mean(x1, axis = -1, keepdims = False) # [B,C,H]
        w = F.mean(w, axis = -1, keepdims = False) # [B,C]
        w = self.linear(w)
        w = F.add_axis(w, axis = -1)
        w = F.add_axis(w, axis = -1)
        x1 = F.concat((x1, F.multiply(x1, w)), axis = 1)  # [B, 2C, H, W]
        del w
        x1 = self.conv2(x1)  # [B, C, H, W]
        return self.lrelu(x + x1)


class CARBBlocks(M.Module):
    def __init__(self, channel_num, block_num):
        super(CARBBlocks, self).__init__()
        self.model = M.Sequential(
            self.make_layer(CARBBlock, channel_num, block_num),
        )

    def make_layer(self, block, channel_num, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(channel_num))
        return M.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PixelShuffle(M.Module):
    def __init__(self, scale=4):
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
        output = F.dimshuffle(output, (0, 1, 4, 3, 5, 2))
        # N C oH oW
        output = output.reshape(N, oC, oH, oW)
        return output


class Identi(M.Module):
    def __init__(self):
        super(Identi, self).__init__()
    
    def forward(self, now_LR, pre_h_SD):
        return pre_h_SD

class SDBlock(M.Module):
    def __init__(self, channel_nums):
        super(SDBlock, self).__init__()
        self.netS = M.Sequential(
            Conv2d(channel_nums, channel_nums, 3, 1, 1),
            M.LeakyReLU(negative_slope=0.05),
            Conv2d(channel_nums, channel_nums, 3, 1, 1)
        )
        self.netD = M.Sequential(
            Conv2d(channel_nums, channel_nums, 3, 1, 1),
            M.LeakyReLU(negative_slope=0.05),
            Conv2d(channel_nums, channel_nums, 3, 1, 1)
        )

    def forward(self, S, D):
        SUM = self.netS(S) + self.netD(D)
        return S + SUM, D + SUM


@BACKBONES.register_module()
class RSDNV2(M.Module):
    """RSDN network structure.

    Paper:
    Ref repo:

    Args:
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 mid_channels = 128,
                 hidden_channels = 3 * 4 * 4,
                 ch = 24,
                 blocknums = 5,
                 upscale_factor=4,
                 hsa = False, 
                 pixel_shuffle = False,
                 window_size = 5):
        super(RSDNV2, self).__init__()
        if hsa: 
            raise NotImplementedError("")
        else:
            self.hsa = Identi()
        self.window_size = window_size
        self.blocknums = blocknums
        self.hidden_channels = hidden_channels

        # 每个LR搞三个尺度(同时适用于S和D)
        self.feature_encoder_carb = M.Sequential(
            M.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1),
            M.LeakyReLU(negative_slope=0.05),
            CARBBlocks(channel_num=ch, block_num=blocknums)
        )
        self.fea_L1_conv1 = M.Conv2d(ch, ch, 3, 2, 1)
        self.fea_L1_conv2 = M.Conv2d(ch, ch, 3, 1, 1)
        self.fea_L2_conv1 = M.Conv2d(ch, ch, 3, 2, 1)
        self.fea_L2_conv2 = M.Conv2d(ch, ch, 3, 1, 1)
        self.lrelu = M.LeakyReLU(negative_slope=0.05)

        self.AU0 = AU(ch = ch) 
        self.AU1 = AU(ch = ch)
        self.AU2 = AU(ch = ch)
        self.UP0 = M.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1)
        self.UP1 = M.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1)

        SDBlocks = []
        for _ in range(blocknums):
            SDBlocks.append(SDBlock(mid_channels))
        self.SDBlocks = M.Sequential(*SDBlocks)
        
        self.pre_SD_S = M.Sequential(
            Conv2d(hidden_channels*2 + self.window_size*2*ch, mid_channels, 3, 1, 1),
            M.LeakyReLU()
        )
        self.pre_SD_D = M.Sequential(
            Conv2d(hidden_channels*2 + self.window_size*2*ch, mid_channels, 3, 1, 1),
            M.LeakyReLU()
        )
        self.conv_SD = M.Sequential(
            Conv2d(mid_channels, hidden_channels, 3, 1, 1),
            M.LeakyReLU()
        )
        self.convS = Conv2d(mid_channels, hidden_channels, 3, 1, 1)
        self.convD = Conv2d(mid_channels, hidden_channels, 3, 1, 1)
        self.convHR = Conv2d(2 * hidden_channels, hidden_channels, 3, 1, 1)

        if pixel_shuffle:
            self.trans_S = PixelShuffle(upscale_factor)
            self.trans_D = PixelShuffle(upscale_factor)
            self.trans_HR = PixelShuffle(upscale_factor)
        else:
            self.trans_S = ConvTranspose2d(hidden_channels, 3, 4, 4, 0, bias=False)
            self.trans_D = ConvTranspose2d(hidden_channels, 3, 4, 4, 0, bias=False)
            self.trans_HR = ConvTranspose2d(hidden_channels, 3, 4, 4, 0, bias=False)

    def deal_before_SD_block(self, x):
        B, N, C, H, W = x.shape  # N video frames
        mid = self.window_size // 2
        
        # L0
        L0_fea = self.feature_encoder_carb(x.reshape(-1, C, H, W))
        # L1
        L1_fea = self.lrelu(self.fea_L1_conv1(L0_fea))
        L1_fea = self.lrelu(self.fea_L1_conv2(L1_fea))
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))

        L0_fea = L0_fea.reshape(B, N, -1, H, W)
        L1_fea = L1_fea.reshape(B, N, -1, H // 2, W // 2)
        L2_fea = L2_fea.reshape(B, N, -1, H // 4, W // 4)
        
        align_LRs = []
        for i in range(N):
            AU2_out = self.AU2(L2_fea[:, mid, ...], L2_fea[:, i, ...])
            AU2_out = self.UP0(AU2_out)
            AU1_out = self.AU1(L1_fea[:, mid, ...], L1_fea[:, i, ...])
            AU1_out = self.UP1(AU1_out + AU2_out)
            del AU2_out
            AU0_out = self.AU0(L0_fea[:, mid, ...], L0_fea[:, i, ...])
            AU0_out = AU0_out + AU1_out
            align_LRs.append(AU0_out)

        align_LRs = F.concat(align_LRs, axis = 1)
        return align_LRs  # [B, 5*48, H, W]

    def forward(self, LR, LR_S, LR_D, pre_S_hat, pre_D_hat, pre_SD):
        """
            args:
            LR: the LR images for this time stamp (5 frames)
            return: 
        """
        pre_SD = self.hsa(LR, pre_SD)  # auto select for hidden SD
        # do mucan 
        LR = self.deal_before_SD_block(LR)  # [B, 5*48, H, W]
        LR_S = self.deal_before_SD_block(LR_S)  # [B, 5*48, H, W]
        LR_D = self.deal_before_SD_block(LR_D)  # [B, 5*48, H, W]
               
        S = F.concat([LR, LR_S, pre_S_hat, pre_SD], axis = 1)
        S = self.pre_SD_S(S)
        D = F.concat([LR, LR_D, pre_D_hat, pre_SD], axis = 1) 
        D = self.pre_SD_D(D)
        for i in range(self.blocknums):
            S,D = self.SDBlocks[i](S, D)
        pre_SD = self.conv_SD(S+D)
        S = self.convS(S)
        D = self.convD(D)
        I = self.convHR(F.concat([S, D], axis=1))
        return self.trans_HR(I), pre_SD, S, D, self.trans_S(S), self.trans_D(D)

    def init_weights(self, pretrained=None, strict=True):
        # 这里也可以进行参数的load，比如不在之前保存的路径中的模型（预训练好的）
        pass
        # """Init weights for models.
        #
        # Args:
        #     pretrained (str, optional): Path for pretrained weights. If given None, pretrained weights will not be loaded. Defaults to None.
        #     strict (boo, optional): Whether strictly load the pretrained model.
        #         Defaults to True.
        # """
        # if isinstance(pretrained, str):
        #     load_checkpoint(self, pretrained, strict=strict, logger=logger)
        # elif pretrained is None:
        #     pass  # use default initialization
        # else:
        #     raise TypeError('"pretrained" must be a str or None. '
        #                     f'But received {type(pretrained)}.')

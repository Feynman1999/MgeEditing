import numpy as np
import megengine
import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
from edit.models.builder import BACKBONES
from edit.models.common import add_H_W_Padding


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

class HSA(M.Module):
    def __init__(self, K):
        super(HSA, self).__init__()
        self.K = K
        self.conv = M.Sequential(
            Conv2d(3, K**2, 3, 1, 1),
            M.ReLU()
        )   
        
    def forward(self, now_LR, pre_h_SD):
        """
            now_LR: B,3,H,W
            pre_h_SD: B,64,H,W
        """
        pad = self.K // 2
        batch, C, H, W = pre_h_SD.shape
        kernels = self.conv(now_LR)  # [B, k*k, H, W]
        # 对 pre_h_SD进行padding
        similarity_matrix = F.zeros_like(pre_h_SD)
        pre_h_SD = add_H_W_Padding(pre_h_SD, margin = pad)
        for i in range(self.K):
            for j in range(self.K):
                # 做点乘
                kernel = kernels[:, i*self.K + j, :, :] # [B, H, W]
                kernel = F.add_axis(kernel, axis = 1)  # [B, 1 ,H, W]
                kernel = F.broadcast_to(kernel, [batch, C, H, W])
                corr = kernel * pre_h_SD[:, :, i:(H+i), j:(W + j)]
                similarity_matrix = similarity_matrix + corr # [B, C, H, W]
        
        similarity_matrix = F.sigmoid(similarity_matrix)
        return F.multiply(pre_h_SD[:, :, pad:(H+pad), pad:(W+pad)], similarity_matrix)


class SDBlock(M.Module):
    def __init__(self, channel_nums):
        super(SDBlock, self).__init__()
        self.netS = M.Sequential(
            Conv2d(channel_nums, channel_nums, 3, 1, 1),
            M.ReLU(),
            Conv2d(channel_nums, channel_nums, 3, 1, 1)
        )
        self.netD = M.Sequential(
            Conv2d(channel_nums, channel_nums, 3, 1, 1),
            M.ReLU(),
            Conv2d(channel_nums, channel_nums, 3, 1, 1)
        )

    def forward(self, S, D):
        SUM = self.netS(S) + self.netD(D)
        return S + SUM, D + SUM


@BACKBONES.register_module()
class RSDN(M.Module):
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
                 blocknums = 5,
                 upscale_factor=4,
                 hsa = False, 
                 pixel_shuffle = False):
        super(RSDN, self).__init__()
        if hsa: 
            self.hsa = HSA(3)
        else:
            self.hsa = Identi()
        self.blocknums = blocknums
        self.hidden_channels = hidden_channels
        SDBlocks = []
        for _ in range(blocknums):
            SDBlocks.append(SDBlock(mid_channels))
        self.SDBlocks = M.Sequential(*SDBlocks)
        
        self.pre_SD_S = M.Sequential(
            Conv2d(2*(3 + hidden_channels), mid_channels, 3, 1, 1),
            M.ReLU(),
        )
        self.pre_SD_D = M.Sequential(
            Conv2d(2*(3 + hidden_channels), mid_channels, 3, 1, 1),
            M.ReLU(),
        )
        self.conv_SD = M.Sequential(
            Conv2d(mid_channels, hidden_channels, 3, 1, 1),
            M.ReLU(),
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

    def forward(self, It, S, D, pre_S, pre_D, pre_S_hat, pre_D_hat, pre_SD):
        """
            args:
            It: the LR image for this time stamp
            S: the structure component of now LR image
            D: the detail component of now LR image
            pre_S: the structure component of pre LR image
            pre_D: the detail component of pre LR image
            pre_S_hat: the hidden state of structure component
            pre_D_hat: the hidden state of detail component
            pre_SD: the overall hidden state

            return: 
        """
        pre_SD = self.hsa(It, pre_SD)  # auto select
        S = F.concat([pre_S, S, pre_S_hat, pre_SD], axis = 1)
        S = self.pre_SD_S(S)
        D = F.concat([pre_D, D, pre_D_hat, pre_SD], axis = 1) 
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

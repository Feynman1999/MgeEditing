import numpy as np
import megengine
import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
from edit.models.builder import BACKBONES
from edit.models.common import compute_cost_volume

class Identi(M.Module):
    def __init__(self):
        super(Identi, self).__init__()
    
    def forward(self, x):
        return x

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
        output = F.dimshuffle(output, (0, 1, 4, 3, 5, 2))
        # N C oH oW
        output = output.reshape(N, oC, oH, oW)
        return output

class CARBBlock(M.Module):
    def __init__(self, channel_num):
        super(CARBBlock, self).__init__()
        self.conv1 = M.Sequential(
            M.Conv2d(channel_num, channel_num, kernel_size=3, padding=1, stride=1),
            M.ReLU(),
            M.Conv2d(channel_num, channel_num, kernel_size=3, padding=1, stride=1),
        )
        # self.global_average_pooling = nn.AdaptiveAvgPool2d((1,1))  # B,C,H,W -> B,C,1,1
        self.linear = M.Sequential(
            M.Linear(channel_num, channel_num // 2),
            M.ReLU(),
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
        w = F.add_axis(w, axis = -1)  # [B,C,1,1]
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


class AU(M.Module):
    def __init__(self, ch):
        super(AU, self).__init__()
        self.conv = M.Sequential(
            M.Conv2d(2*ch, ch, kernel_size=3, stride=1, padding=1),
            M.LeakyReLU(),
            M.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            M.LeakyReLU(),
        )

    def forward(self, mid, ref):
        """
            mid: [B, C, H, W]
            ref: [B, C, H, W]
            return: [B, C, H, W]
        """
        mid = F.concat([mid, ref], axis = 1)
        return self.conv(mid)


class Separate_non_local(M.Module):
    def __init__(self, channel_num, frames):
        super(Separate_non_local, self).__init__()
        self.frames = frames
        self.A2 = M.Conv2d(channel_num * frames, channel_num * frames, kernel_size=1, padding=0, stride=1)
        self.B2 = M.Conv2d(channel_num * frames, channel_num * frames, kernel_size=1, padding=0, stride=1)
        self.D2 = M.Conv2d(channel_num * frames, channel_num * frames, kernel_size=1, padding=0, stride=1)
        self.A3 = M.Conv2d(channel_num * frames, channel_num * frames, kernel_size=1, padding=0, stride=1)
        self.B3 = M.Conv2d(channel_num * frames, channel_num * frames, kernel_size=1, padding=0, stride=1)
        self.D3 = M.Conv2d(channel_num * frames, channel_num * frames, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = self.frames
        C = C // N
        A2 = F.dimshuffle(self.A2(x).reshape(B, N, C, H, W), (0, 2, 1, 3, 4)).reshape(B, C, N*H*W)
        B2 = F.dimshuffle(self.B2(x).reshape(B, N, C, H, W), (0, 1, 3, 4, 2)).reshape(B, N*H*W, C)
        A3 = self.A3(x).reshape(B, N, C, H, W).reshape(B, N, C*H*W)
        B3 = F.dimshuffle(self.B3(x).reshape(B, N, C, H, W).reshape(B, N, C*H*W), (0, 2, 1))

        D2 = F.dimshuffle(self.D2(x).reshape(B, N, C, H, W), (0, 2, 1, 3, 4)).reshape(B, C, N*H*W)
        D3 = self.D3(x).reshape(B, N, C, H, W).reshape(B, N, C*H*W)

        attention2 = F.softmax(F.batched_matrix_mul(A2, B2), axis = -1)  # [B, C, C]
        attention3 = F.softmax(F.batched_matrix_mul(A3, B3), axis = -1)  # [B, N, N]

        E2 = F.dimshuffle(F.batched_matrix_mul(attention2, D2).reshape(B, C, N, H, W), (0, 2, 1, 3, 4)).reshape(B, N*C, H, W)
        E3 = F.batched_matrix_mul(attention3, D3).reshape(B, N*C, H, W)
        return x + E2 + E3


@BACKBONES.register_module()
class MUCANV2(M.Module):
    """MUCAN network structure.

    Paper:
    Ref repo:

    Args:
    """
    def __init__(self,
                 ch=128,
                 nframes = 7,
                 input_nc = 3,
                 output_nc = 3,
                 upscale_factor=4,
                 blocknums1 = 5,
                 blocknums2 = 15,
                 non_local = True):
        super(MUCANV2, self).__init__()
        self.nframes = nframes
        self.upscale_factor = upscale_factor
        # 每个LR搞三个尺度
        self.feature_encoder_carb = M.Sequential(
            M.Conv2d(input_nc, ch, kernel_size=3, stride=1, padding=1),
            M.LeakyReLU(negative_slope=0.05),
            CARBBlocks(channel_num=ch, block_num=blocknums1)
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

        if non_local:
            self.non_local = Separate_non_local(ch, nframes)
        else:
            self.non_local = Identi()

        self.aggre = M.Conv2d(ch * self.nframes, ch, kernel_size=3, stride=1, padding=1)

        self.carbs = M.Sequential(
            CARBBlocks(channel_num=ch, block_num=blocknums2),
        )

        self.main_conv = M.Sequential(
            M.Conv2d(ch, ch*4, kernel_size=3, stride=1, padding=1),
            M.LeakyReLU(),
            PixelShuffle(scale=2), # 128
            M.Conv2d(ch, ch*2, kernel_size=3, stride=1, padding=1),
            M.LeakyReLU(),
            PixelShuffle(scale=2),  # 64
            M.Conv2d(ch//2, ch//2, kernel_size=3, stride=1, padding=1),
            M.LeakyReLU(),
            M.Conv2d(ch//2, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, x_mid_bicubic):
        """
        :param x: e.g. [4, 7, 3, 64, 64]
        :return:
        """
        B, N, C, H, W = x.shape  # N video frames
        mid = self.nframes // 2
        
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

        align_LRs = F.concat(align_LRs, axis = 1)  # [B, N*C, H, W]
        align_LRs = self.non_local(align_LRs)  # [B, N*C, H, W]
        align_LRs = self.aggre(align_LRs)
        align_LRs = align_LRs + self.carbs(align_LRs)
        return self.main_conv(align_LRs) + x_mid_bicubic

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

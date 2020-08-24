import numpy as np
import megengine
import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
from edit.models.builder import BACKBONES
from .dbpn import UPU, DPU

class ResBlock(M.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='prelu'):
        super(ResBlock, self).__init__()

        if activation == 'relu':
            self.act = M.ReLU()
        elif activation == 'prelu':
            self.act = M.PReLU(num_parameters=1, init=0.25)
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
    def __init__(self, channel_num, resblock_num, kernel_size=3, activation='prelu'):
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

class SISR_Block(M.Module):
    def __init__(self, cl, ch):
        super(SISR_Block, self).__init__()
        self.num_stages = 3
        self.pre_deal = M.Conv2d(cl, ch, kernel_size=1, stride=1, padding=0)
        self.prelu = M.PReLU(num_parameters=1, init=0.25)
        self.UPU1 = UPU(ch, 8, stride=4, padding=2)
        self.UPU2 = UPU(ch, 8, stride=4, padding=2)
        self.UPU3 = UPU(ch, 8, stride=4, padding=2)
        self.DPU1 = DPU(ch, 8, stride=4, padding=2)
        self.DPU2 = DPU(ch, 8, stride=4, padding=2)
        self.reconstruction = M.Conv2d(self.num_stages * ch, ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.prelu(self.pre_deal(x))

        h1 = self.UPU1(x)
        h2 = self.UPU2(self.DPU1(h1))
        h3 = self.UPU3(self.DPU2(h2))
        
        x = self.reconstruction(F.concat((h3, h2, h1), axis=1))
        return x


class Residual_Blocks(M.Module):
    def __init__(self, ch):
        super(Residual_Blocks, self).__init__()
        self.model = M.Sequential(
            ResBlocks(channel_num=ch, resblock_num=5, kernel_size=3),
            M.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            M.PReLU(),
        )

    def forward(self, x):
        return self.model(x)


class MISR_Block(M.Module):
    def __init__(self, cm, ch):
        super(MISR_Block, self).__init__()
        self.model = M.Sequential(
            ResBlocks(channel_num=cm, resblock_num=5, kernel_size=3),
            M.ConvTranspose2d(cm, ch, kernel_size=8, stride=4, padding=2),
            M.PReLU(),
        )

    def forward(self, x):
        return self.model(x)


class Decoder(M.Module):
    def __init__(self, ch, cl):
        super(Decoder, self).__init__()
        self.model = M.Sequential(
            ResBlocks(channel_num=ch, resblock_num=5, kernel_size=3),
            M.Conv2d(ch, cl, kernel_size=8, stride=4, padding=2),
            M.PReLU(),
        )

    def forward(self, x):
        return self.model(x)


class Projection_Module(M.Module):
    def __init__(self, cl, cm ,ch):
        super(Projection_Module, self).__init__()
        self.misr = MISR_Block(cm, ch)
        self.sisr = SISR_Block(cl, ch)
        self.res = Residual_Blocks(ch)
        self.decoder = Decoder(ch, cl)

    def forward(self, M, L):
        hm = self.misr(M)
        hl = self.sisr(L)
        hm = hl - hm  # error
        hm = self.res(hm)
        hl = hl + hm
        next_l = self.decoder(hl)
        return hl, next_l


@BACKBONES.register_module()
class RBPN(M.Module):
    """RBPN network structure.

    Paper:
    Ref repo:

    Args:
    """
    def __init__(self,
                 cl=32,
                 cm=32,
                 ch=16,
                 nframes = 7,
                 input_nc = 3,
                 output_nc = 3,
                 upscale_factor=4):
        super(RBPN, self).__init__()
        self.nframes = nframes
        self.upscale_factor = upscale_factor
        #Initial Feature Extraction
        self.conv1 = M.Sequential(
            M.Conv2d(input_nc, cl, kernel_size=3, stride=1, padding=1),
            M.PReLU(),
        )
        self.conv2 = M.Sequential(
            M.Conv2d(input_nc*2+0, cm, kernel_size=3, stride=1, padding=1),
            M.PReLU(),
        )
        # projection module
        self.Projection = Projection_Module(cl, cm, ch)

        # reconstruction module
        self.reconstruction = M.Conv2d((self.nframes-1)*ch, output_nc, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        :param x: e.g. [4, 7, 3, 64, 64]
        :return:
        """
        mid = self.nframes // 2
        # [0, mid)  [mid+1, nframes)
        L = self.conv1(x[:, mid, ...])
        Hlist = []
        for idx in range(self.nframes):
            if idx == mid:
                continue
            M = self.conv2(F.concat((x[:, mid, ...], x[:, idx, ...]), axis=1))
            H, L = self.Projection(M, L)
            Hlist.append(H)

        return self.reconstruction(F.concat(Hlist, axis=1))

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

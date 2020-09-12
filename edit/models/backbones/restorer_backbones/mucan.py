import numpy as np
import megengine
import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
from edit.models.builder import BACKBONES
from edit.models.common import compute_cost_volume

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


class AU_CV(M.Module):
    def __init__(self, K, d, ch):
        super(AU_CV, self).__init__()
        self.K = K
        self.d = d
        self.conv = M.Sequential(
            M.Conv2d(ch * self.K, ch, kernel_size=3, stride=1, padding=1),
            M.LeakyReLU()
        )

    def forward(self, mid, ref):
        B, C, H, W = mid.shape
        mid = F.normalize(mid, p=2, axis = 1)
        ref = F.normalize(ref, p=2, axis = 1)
        cost_volume, ref = compute_cost_volume(mid, ref, max_displacement = self.d)  # [B, (2d+1)**2, H, W]
        cost_volume = F.dimshuffle(cost_volume, (0, 2, 3, 1))
        cost_volume = cost_volume.reshape((-1, (2*self.d + 1)**2))
        # argmax
        indices = F.top_k(cost_volume, k = self.K, descending = True)[1]  # [B*H*W, K]
        del cost_volume
        ref_list = [] # [B, C, H, W]
        origin_i_j = F.arange(0, H*W, 1)  # float32
        origin_i = F.floor(origin_i_j / W)  # (H*W, ) 
        origin_j = F.mod(origin_i_j, W)  # (H*W, )
        del origin_i_j
        # reshape ref
        ref = ref.reshape((B, C, (H+2*self.d)*(W+2*self.d)))
        for i in range(self.K):
            index = indices[:, i]  # [B*H*W, ]
            index = index.reshape((-1, H*W))
            index_i = F.floor(index / (2*self.d + 1)) + origin_i  # [B, H*W] 
            index_j = F.mod(index, (2*self.d + 1)) + origin_j  # [B, H*W]
            # 根据每个pixel的i,j 算出index
            index = index_i * W + index_j  # [B, H*W]
            index = index.astype('int32')
            # add axis
            index=  F.add_axis(index, axis = 1) # [B, 1, H*W]
            # broadcast
            index = F.broadcast_to(index, (B, C, H*W))
            # gather
            output = F.gather(ref, axis = 2, index = index)  # [B, C, H*W]
            ref_list.append(output.reshape((B, C, H, W)))
        return self.conv(F.concat(ref_list, axis = 1))


@BACKBONES.register_module()
class MUCAN(M.Module):
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
                 use_cost_volume = False):
        super(MUCAN, self).__init__()
        self.nframes = nframes
        self.upscale_factor = upscale_factor
        # 每个LR搞三个尺度
        self.feature_encoder_carb = M.Sequential(
            M.Conv2d(input_nc, ch, kernel_size=3, stride=1, padding=1),
            M.LeakyReLU(negative_slope=0.05),
            CARBBlocks(channel_num=ch, block_num=4)
        )
        self.fea_L1_conv1 = M.Conv2d(ch, ch, 3, 2, 1)
        self.fea_L1_conv2 = M.Conv2d(ch, ch, 3, 1, 1)
        self.fea_L2_conv1 = M.Conv2d(ch, ch, 3, 2, 1)
        self.fea_L2_conv2 = M.Conv2d(ch, ch, 3, 1, 1)
        self.lrelu = M.LeakyReLU(negative_slope=0.05)
        
        if use_cost_volume:
            self.AU0 = AU_CV(K=6, d = 5, ch = ch) 
            self.AU1 = AU_CV(K=5, d = 3, ch = ch)
            self.AU2 = AU_CV(K=4, d = 3, ch = ch)
        else:
            self.AU0 = AU(ch = ch) 
            self.AU1 = AU(ch = ch)
            self.AU2 = AU(ch = ch)

        self.UP0 = M.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1)
        self.UP1 = M.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1)

        self.aggre = M.Conv2d(ch * self.nframes, ch, kernel_size=3, stride=1, padding=1)

        self.main_conv = M.Sequential(
            CARBBlocks(channel_num=ch, block_num=10),
            M.ConvTranspose2d(ch, ch//2, kernel_size=4, stride=2, padding=1),
            M.LeakyReLU(),
            M.Conv2d(ch//2, ch//2, kernel_size=3, stride=1, padding=1),
            M.LeakyReLU(),
            M.ConvTranspose2d(ch//2, ch//4, kernel_size=4, stride=2, padding=1),
            M.LeakyReLU(),
            M.Conv2d(ch//4, output_nc, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        """
        :param x: e.g. [4, 5, 3, 64, 64]
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

        align_LRs = F.concat(align_LRs, axis = 1)
        align_LRs = self.aggre(align_LRs)
        return self.main_conv(align_LRs)

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

import numpy as np
import megengine
import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
from edit.models.common import ResBlocks, ShuffleV2Block, MobileNeXt, default_init_weights, PixelShufflePack
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

def backwarp(tenInput, tenFlow, border_mode):
    """
        CONSTANT(0)    REPLICATE
    """
    _, _, H, W = tenFlow.shape
    if str(tenFlow.shape) not in backwarp_tenGrid.keys():
        x_list = np.linspace(0., W - 1., W).reshape(1, 1, 1, W)
        x_list = x_list.repeat(H, axis=2)
        y_list = np.linspace(0., H - 1., H).reshape(1, 1, H, 1)
        y_list = y_list.repeat(W, axis=3)
        xy_list = np.concatenate((x_list, y_list), 1)  # [1,2,H,W]
        backwarp_tenGrid[str(tenFlow.shape)] = megengine.tensor(xy_list.astype(np.float32))
    return F.nn.remap(inp = tenInput, map_xy=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).transpose(0, 2, 3, 1), border_mode=border_mode)

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
    def __init__(self, num_layers, pretrain_ckpt_path = None, blocktype = None):
        super(Spynet, self).__init__()
        assert num_layers in (4, 5, 6)
        self.num_layers = num_layers
        self.threshold = 2
        self.pretrain_ckpt_path = pretrain_ckpt_path
        self.blocktype = blocktype

        if self.blocktype == "resblock" or self.blocktype == "RK4":
            basic_list = [ Basic(intLevel) for intLevel in range(num_layers) ]
            self.border_mode = "REPLICATE"
        elif self.blocktype == "shuffleblock":
            basic_list = [ Basic_Shuffle(intLevel) for intLevel in range(num_layers) ]
            self.border_mode = "REPLICATE"
        elif self.blocktype == "MobileNeXt":
            basic_list = [ Basic_CA(intLevel) for intLevel in range(num_layers) ]
            self.border_mode = "REPLICATE"
        self.netBasic = M.Sequential(*basic_list)

    def preprocess(self, tenInput):
        tenRed = (tenInput[:, 0:1, :, :] - 0.485) / 0.229
        tenGreen = (tenInput[:, 1:2, :, :] - 0.456) / 0.224
        tenBlue = (tenInput[:, 2:3, :, :] - 0.406 ) / 0.225
        return F.concat([tenRed, tenGreen, tenBlue], axis=1) # [B,3,H,W]

    def cal_resize_value(self, x, resize_multi = 32):
        diff = x % resize_multi
        if diff == 0:
            return x
        return x + resize_multi - diff

    def forward(self, tenFirst, tenSecond):
        # 如果输入的图片不是32的倍数则进行resize
        _,_,H,W = tenFirst.shape
        resize_flag = False
        if H % 32 != 0 or W % 32 != 0:
            resize_flag = True
            # 算出需要resize到的大小
            aim_H = self.cal_resize_value(H)
            aim_W = self.cal_resize_value(W)
            tenFirst = F.nn.interpolate(tenFirst, size=[aim_H, aim_W], align_corners=False)
            tenSecond = F.nn.interpolate(tenSecond, size=[aim_H, aim_W], align_corners=False)

        tenFirst = [self.preprocess(tenFirst)]
        tenSecond = [self.preprocess(tenSecond)]

        for intLevel in range(self.num_layers - 1):
            if tenFirst[0].shape[2] >= self.threshold or tenFirst[0].shape[3] >= self.threshold:
                tenFirst.insert(0, F.avg_pool2d(inp=tenFirst[0], kernel_size=2, stride=2))
                tenSecond.insert(0, F.avg_pool2d(inp=tenSecond[0], kernel_size=2, stride=2))
        
        tenFlow = F.zeros([tenFirst[0].shape[0], 2, tenFirst[0].shape[2], tenFirst[0].shape[3]])
        tenUpsampled = tenFlow
        tenFlow = self.netBasic[0]( F.concat([tenFirst[0], backwarp(tenInput=tenSecond[0], tenFlow=tenUpsampled, border_mode=self.border_mode), tenUpsampled], axis=1) ) + tenUpsampled
        for intLevel in range(1, len(tenFirst)):
            # large:   6 for training  (2*2, 4*4, 8*8, 16*16, 32*32, 64*64)
            # middle:  4 for training  (8*8, 16*16, 32*32, 64*64)  4 for test (24*40, 48*80, 96*160, 192*320)
            tenUpsampled = F.nn.interpolate(inp=tenFlow, scale_factor=2, mode='BILINEAR', align_corners=True) * 2.0
            tenFlow = self.netBasic[intLevel]( F.concat([tenFirst[intLevel], backwarp(tenInput=tenSecond[intLevel], tenFlow=tenUpsampled, border_mode=self.border_mode), tenUpsampled], axis=1) ) + tenUpsampled
        
        # adjust the flow values
        if resize_flag:
            tenFlow = F.nn.interpolate(inp=tenFlow, size=[H, W], align_corners=False)
            tenFlow[:, 0, :, :] *= float(W) / float(aim_W)
            tenFlow[:, 1, :, :] *= float(H) / float(aim_H)

        return tenFlow

    def init_weights(self, strict=True):
        if self.pretrain_ckpt_path is not None:
            print("loading pretrained model for Spynet 🤡🤡🤡🤡🤡🤡...")
            state_dict = megengine.load(self.pretrain_ckpt_path)
            self.load_state_dict(state_dict, strict=strict)
        else:
            default_init_weights(self.netBasic, scale=0.2)

class ONE_LAYER(M.Module):
    def __init__(self, hidden_channels, blocknums, blocktype = "resblock", use_flow_mask = False):
        super(ONE_LAYER, self).__init__()
        self.hidden_channels = hidden_channels
        self.use_flow_mask = use_flow_mask

        self.feature_extracter = ResBlocks(channel_num=hidden_channels, resblock_num=blocknums, blocktype=blocktype)
        
        if use_flow_mask:
            self.border_mode = "CONSTANT" # 实际上用什么模式都可以，因为都会被mask掉，所以用速度快的即可
        else:
            self.border_mode = "REPLICATE"
            
        # self.shift_pixels = 3
        # self.is_shift = is_shift
        # self.directions = [[-1,-1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]
        self.init_weights()

    def forward(self, now_hidden, ref_hidden, flow, mask, reduction):
        # 会使用两次，ref_hidden分别是forward hidden和backward hidden
        # flow: [B,2,H,W]
        # mask: [B,1,H,W]
        ref_hidden = backwarp(ref_hidden, flow, self.border_mode) # [B, C, H, W]

        if self.use_flow_mask:
            ref_hidden = (1-mask)*now_hidden + mask * ref_hidden
        
        now_hidden = reduction(F.concat([now_hidden, ref_hidden], axis=1))
        now_hidden = self.feature_extracter(now_hidden)
        return now_hidden

    def init_weights(self):
        pass

@BACKBONES.register_module()
class BasicVSR_MULTI_LAYER(M.Module):
    def __init__(self, in_channels, 
                        out_channels, 
                        hidden_channels,
                        blocknums, 
                        reconstruction_blocks, 
                        upscale_factor, 
                        pretrained_optical_flow_path, 
                        flownet_layers = 5,
                        blocktype = "resblock",
                        RNN_layers = 4,
                        use_flow_mask = False):
        super(BasicVSR_MULTI_LAYER, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.blocknums = blocknums
        self.upscale_factor = upscale_factor
        self.reconstruction_blocknums = reconstruction_blocks
        self.RNN_layers = RNN_layers

        self.flownet = Spynet(num_layers=flownet_layers, pretrain_ckpt_path=pretrained_optical_flow_path, blocktype = blocktype)

        self.conv_rgb = M.ConvRelu2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1) # need init
        self.reduction_warp = M.Sequential(
            M.Conv2d(2*hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            M.LeakyReLU(negative_slope=0.1)
        )
        self.reduction_fb = M.Sequential(
            M.Conv2d(2*hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            M.LeakyReLU(negative_slope=0.1)
        )

        layers = []
        for _ in range(RNN_layers):
            layers.append(ONE_LAYER(hidden_channels = hidden_channels, blocknums = blocknums, blocktype = blocktype, use_flow_mask=use_flow_mask))
        self.layers = M.Sequential(*layers)

        self.reconstruction = ResBlocks(channel_num=hidden_channels, resblock_num=reconstruction_blocks, blocktype=blocktype)
        self.upsample1 = PixelShufflePack(hidden_channels, hidden_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(hidden_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = M.Conv2d(64, 64, 3, 1, 1)  # need init
        self.conv_last = M.Conv2d(64, out_channels, 3, 1, 1)
        
        self.lrelu = M.LeakyReLU(negative_slope=0.1)

    def do_upsample(self, now_hidden):
        out = self.reconstruction(now_hidden)
        out = self.lrelu(self.upsample1(out))
        out = self.lrelu(self.upsample2(out))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        return out

    def rgb_feature(self, x):
        return self.conv_rgb(x)

    def forward(self, now_hidden, ref_hidden, flow, layer_index, mask = None):
        return self.layers[layer_index](now_hidden, ref_hidden, flow, mask, self.reduction_warp)

    def aggr_forward_backward_hidden(self, forward_hidden, backward_hidden, layer_index):
        return self.reduction_fb(F.concat([forward_hidden, backward_hidden], axis=1))

    def init_weights(self, pretrained):
        self.flownet.init_weights(strict=False)
        default_init_weights(self.conv_rgb)
        default_init_weights(self.reduction_fb, scale = 0.1, nonlinearity='leaky_relu', lrelu_value=0.1)
        default_init_weights(self.reduction_warp, scale = 0.1, nonlinearity='leaky_relu', lrelu_value=0.1)
        default_init_weights(self.upsample1, nonlinearity='leaky_relu', lrelu_value=0.1)
        default_init_weights(self.upsample2, nonlinearity='leaky_relu', lrelu_value=0.1)
        default_init_weights(self.conv_hr, nonlinearity='leaky_relu', lrelu_value=0.1)

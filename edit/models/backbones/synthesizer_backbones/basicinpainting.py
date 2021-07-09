from megengine.functional.nn import sigmoid
import numpy as np
import megengine
import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
from edit.models.common import ResBlocks, ShuffleV2Block, MobileNeXt, default_init_weights, PixelShufflePack
from edit.models.builder import BACKBONES
import math

backwarp_tenGrid = {}

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
    def __init__(self, num_layers, pretrain_ckpt_path = None):
        super(Spynet, self).__init__()
        assert num_layers in (4, 5, 6)
        self.num_layers = num_layers
        self.threshold = 2
        self.pretrain_ckpt_path = pretrain_ckpt_path

        basic_list = [ Basic(intLevel) for intLevel in range(num_layers) ]
        self.border_mode = "REPLICATE"
        self.netBasic = M.Sequential(*basic_list)

    def preprocess(self, tenInput):
        tenRed = (tenInput[:, 0:1, :, :]*0.5 + 0.5 - 0.485) / 0.229
        tenGreen = (tenInput[:, 1:2, :, :]*0.5 + 0.5 - 0.456) / 0.224
        tenBlue = (tenInput[:, 2:3, :, :]*0.5 + 0.5 - 0.406 ) / 0.225
        return F.concat([tenRed, tenGreen, tenBlue], axis=1) # [B,3,H,W]

    def forward(self, tenFirst, tenSecond):
        # 输入的图片统一resize到72, 128
        _,_,H,W = tenFirst.shape
        aim_H = 96
        aim_W = 128
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
        
        tenFlow = F.nn.interpolate(inp=tenFlow, size=[72, 128], align_corners=False)
        tenFlow[:, 1, :, :] *= float(72) / float(aim_H)

        return tenFlow

    def init_weights(self, strict=True):
        if self.pretrain_ckpt_path is not None:
            print("loading pretrained model for Spynet 🤡🤡🤡🤡🤡🤡...")
            state_dict = megengine.load(self.pretrain_ckpt_path)
            self.load_state_dict(state_dict, strict=strict)
        else:
            default_init_weights(self.netBasic, scale=0.2)

class GatedResBlock(M.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, init_scale = 0.1):
        super(GatedResBlock, self).__init__()
        self.init_scale = init_scale
        self.conv1 = M.ConvRelu2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size//2))
        self.conv2 = M.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size//2))
        self.conv3 = M.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size//2))
        self.init_weights()

    def init_weights(self):
        default_init_weights(self.conv1, scale=self.init_scale)
        default_init_weights(self.conv2, scale=self.init_scale)
        default_init_weights(self.conv3, scale=self.init_scale)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        mask = self.conv2(x)
        out = F.sigmoid(mask) * out
        out = self.conv3(out)
        return identity + out

class GatedResBlocks(M.Module):
    def __init__(self, channel_num, resblock_num, kernel_size=3):
        super(GatedResBlocks, self).__init__()
        self.model = M.Sequential(
            self.make_block_layer(channel_num, resblock_num, kernel_size),
        )
        
    def make_block_layer(self, ch_out, num_blocks, kernel_size):
        layers = []
        for _ in range(num_blocks):
            layers.append(GatedResBlock(ch_out, ch_out, kernel_size))
        return M.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ONE_LAYER(M.Module):
    def __init__(self, type, hidden_channels, blocknums, use_flow_mask = False, have_skip_connect = False):
        super(ONE_LAYER, self).__init__()
        self.hidden_channels = hidden_channels
        self.use_flow_mask = use_flow_mask
        self.have_skip_connect = have_skip_connect
        assert type in (1, 2, 3)

        if have_skip_connect:
            self.reduction = M.ConvRelu2d(3*hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1) # need init
        else:
            self.reduction = M.ConvRelu2d(2*hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1) # need init

        self.feature_extracter = GatedResBlocks(channel_num=hidden_channels, resblock_num=blocknums)

        if type == 1:
            self.feature_output = M.ConvRelu2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1) # need init
        elif type == 2:
            self.feature_output = M.ConvRelu2d(hidden_channels, 2*hidden_channels, kernel_size=3, stride=2, padding=1) # need init
        else:
            self.feature_output = M.Sequential(
                PixelShufflePack(hidden_channels, hidden_channels//2, scale_factor = 2, upsample_kernel=3),
                M.ReLU()
            )

        if use_flow_mask:
            self.border_mode = "CONSTANT" # 实际上用什么模式都可以，因为都会被mask掉，所以用速度快的即可
        else:
            self.border_mode = "REPLICATE"
            
        self.init_weights()

    def forward(self, now_hidden, ref_hidden, flow, mask = None):
        # flow: [B,2,H,W]
        # mask: [B,1,H,W]
        # ref_hidden = backwarp(ref_hidden, flow, self.border_mode) # [B, C, H, W]
        # if self.use_flow_mask:
        #    ref_hidden = (1-mask)*now_hidden + mask * ref_hidden
        now_hidden = self.reduction(F.concat([now_hidden, ref_hidden], axis=1))
        this_layer = self.feature_extracter(now_hidden)
        return this_layer

    def init_weights(self):
        default_init_weights(self.reduction, scale=0.1)
        default_init_weights(self.feature_output, scale=0.1)

@BACKBONES.register_module()
class BASIC_Inpaint(M.Module):
    def __init__(self,
                reconstruction_blocks, #　at last
                pretrained_optical_flow_path, 
                blocknums = 3, # one layer, every block start with a gated conv, thus total blocknum * 3  convs for one layer
                flownet_layers = 5):
        super(BASIC_Inpaint, self).__init__()
        self.blocknums = blocknums
        self.reconstruction_blocknums = reconstruction_blocks

        self.flownet = Spynet(num_layers=flownet_layers, pretrain_ckpt_path=pretrained_optical_flow_path)

        self.conv0 = M.Conv2d(4, 32, 3, 2, 1)
        self.conv1 = M.Sequential(
            M.Conv2d(32, 32, 3, 1, 1),
            M.LeakyReLU(0.1),
        )
        
        layers = []
        layers.append(ONE_LAYER(2, 32, blocknums, use_flow_mask=False, have_skip_connect=False))
        layers.append(ONE_LAYER(2, 64, blocknums, use_flow_mask=False, have_skip_connect=False))
        layers.append(ONE_LAYER(1, 128, blocknums, use_flow_mask=False, have_skip_connect=False))
        layers.append(ONE_LAYER(3, 128, blocknums, use_flow_mask=False, have_skip_connect=True))
        layers.append(ONE_LAYER(3, 64, blocknums, use_flow_mask=False, have_skip_connect=True))
        layers.append(ONE_LAYER(1, 32, blocknums, use_flow_mask=False, have_skip_connect=True))
        self.layers = layers

        self.border_mode = "REPLICATE"
        self.reconstruction = ResBlocks(channel_num=32, resblock_num=reconstruction_blocks)
        self.upsample = PixelShufflePack(32, 16, 2, upsample_kernel=3)
        self.conv_hr = M.Conv2d(16, 16, 3, 1, 1)  # need init
        self.conv_last = M.Conv2d(16, 3, 3, 1, 1)
        self.lrelu = M.LeakyReLU(negative_slope=0.2)

    def forward(x):
        pass

    def do_upsample(self, hidden):
        out = self.reconstruction(hidden)
        out = self.lrelu(self.upsample(out))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        return out # [B, 3, 4*H, 4*W]

    def init_weights(self, pretrained):
        self.flownet.init_weights(strict=False)

        default_init_weights(self.conv1, nonlinearity='leaky_relu', lrelu_value=0.1)
        default_init_weights(self.upsample, nonlinearity='leaky_relu', lrelu_value=0.2)
        default_init_weights(self.conv_hr, nonlinearity='leaky_relu', lrelu_value=0.2)

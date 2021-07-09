"""
    用于mgtv的sttn (baseline)
    区别：attention时, blocksize不同
    加入spynet，用于训练光流
"""
import numpy as np
import megengine
import megengine.module as M
import megengine.functional as F
from megengine.module.conv import Conv2d
from edit.models.builder import BACKBONES
from edit.models.common import default_init_weights
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
        _,_,H,W = tenFirst.shape

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
        
        return tenFlow

    def init_weights(self, strict=True):
        if self.pretrain_ckpt_path is not None:
            print("loading pretrained model for Spynet 🤡🤡🤡🤡🤡🤡...")
            state_dict = megengine.load(self.pretrain_ckpt_path)
            self.load_state_dict(state_dict, strict=strict)
        else:
            default_init_weights(self.netBasic, scale=0.2)


class FeedForward(M.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        self.conv = M.Sequential(
                        M.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
                        M.LeakyReLU(0.2),
                        M.Conv2d(d_model, d_model, kernel_size=3, padding=1),
                        M.LeakyReLU(0.2)
                    )
        self.init_weights()

    def forward(self, x):
        return self.conv(x)

    def init_weights(self):
        default_init_weights(self.conv, scale=0.1, nonlinearity="leaky_relu", lrelu_value = 0.2)

def do_attention(query, key, value, mask):
    # mask = mask.astype("float32")
    # print(query.shape, key.shape)
    scores = F.matmul(query, key.transpose(0, 2, 1)) / math.sqrt(query.shape[-1]) # b, t*out_h*out_w, t*out_h*out_w
    scores[mask] = -1e9 
    # scores = scores * (1-mask) + scores * mask * (-1e9)
    p_attn = F.nn.softmax(scores, axis=2)
    # print(p_attn.shape, value.shape)
    p_val = F.matmul(p_attn, value)
    return p_val, p_attn

class MultiHeadedAttention(M.Module):
    def __init__(self, heads, hidden):
        super(MultiHeadedAttention, self).__init__()
        self.headnums = heads
        self.patchsize = [(64, 36), (32, 18), (16, 9), (4, 6)]
        self.query_embedding = M.Conv2d(hidden, hidden, kernel_size=1, padding=0)
        self.key_embedding = M.Conv2d(hidden, hidden, kernel_size=1, padding=0)
        self.value_embedding = M.Conv2d(hidden, hidden, kernel_size=1, padding=0)
        self.output_linear = M.Sequential(
            M.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            M.LeakyReLU(0.2)
        )
        self.thr = 0.5 # 可以适当放大，这样只有占很多空白的才会是1
        self.init_weights()

    def forward(self, x, m, b, c):
        bt, _, h, w = x.shape
        t = bt // b
        d_k = c // self.headnums  # 每个head的通道数
        outputs = []
        _query = self.query_embedding(x)
        _key = self.key_embedding(x)
        _value = self.value_embedding(x)
        for idx in range(self.headnums):
            width, height = self.patchsize[idx]
            query, key, value = _query[:, (idx*d_k):(idx*d_k + d_k), ...], _key[:, (idx*d_k):(idx*d_k + d_k), ...], _value[:, (idx*d_k):(idx*d_k + d_k), ...]
            out_h, out_w = h // height, w // width
            # 0) mask
            mm = m.reshape(b, t, 1, out_h, height, out_w, width)
            mm = mm.transpose(0, 1, 3, 5, 2, 4, 6).reshape(b, t*out_h*out_w, height*width)
            mm = F.expand_dims((F.mean(mm, axis=2) > self.thr), axis=1) # (b, 1, t*out_h*out_w)
            mm = F.broadcast_to(mm, (b, t*out_h*out_w, t*out_h*out_w))  
            # mm = F.tile(mm, (1, t*out_h*out_w, 1)) # (b, t*out_h*out_w, t*out_h*out_w)， 每一行表示0， 1权重
            # 1) embedding and reshape      
            query = query.reshape(b, t, d_k, out_h, height, out_w, width)
            query = query.transpose((0, 1, 3, 5, 2, 4, 6)).reshape((b,  t*out_h*out_w, d_k*height*width))
            key = key.reshape(b, t, d_k, out_h, height, out_w, width)
            key = key.transpose((0, 1, 3, 5, 2, 4, 6)).reshape((b,  t*out_h*out_w, d_k*height*width))
            value = value.reshape(b, t, d_k, out_h, height, out_w, width)
            value = value.transpose((0, 1, 3, 5, 2, 4, 6)).reshape((b,  t*out_h*out_w, d_k*height*width))
            # 2) Apply attention on all the projected vectors in batch.
            y, _ = do_attention(query, key, value, mm)
            # 3) "Concat" using a view and apply a final linear.
            # print(y.shape)
            # print(b, t, out_h, out_w, d_k, height, width)
            y = y.reshape(b, t, out_h, out_w, d_k, height, width)
            y = y.transpose((0, 1, 4, 2, 5, 3, 6)).reshape(bt, d_k, h, w)
            outputs.append(y)
        outputs = F.concat(outputs, axis = 1)
        return self.output_linear(outputs)

    def init_weights(self):
        default_init_weights(self.output_linear, scale=0.1, nonlinearity="leaky_relu", lrelu_value = 0.2)

class TransformerBlock(M.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """
    def __init__(self, heads, hidden):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads, hidden)
        self.feed_forward = FeedForward(hidden)

    def forward(self, Dict):
        x = Dict['x']
        m = Dict['m']
        b = Dict['b']
        c = Dict['c']
        x = x + self.attention(x, m, b, c)
        x = x + self.feed_forward(x)
        return {'x':x, 'm':m, 'b':b, 'c':c}

class deconv(M.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super(deconv, self).__init__()
        self.conv = M.Conv2d(input_channel, output_channel, kernel_size, 1, padding)

    def forward(self, x):
        x = F.nn.interpolate(x, scale_factor = 2, mode='BILINEAR', align_corners=True)
        return self.conv(x)

@BACKBONES.register_module()
class STTN_MGTV_Flow(M.Module):
    def __init__(self,
                 pretrained_optical_flow_path,
                 channels = 64,
                 layers = 8,
                 heads = 4):
        super(STTN_MGTV_Flow, self).__init__()

        self.channels = channels * heads
        self.flownet = Spynet(num_layers=5, pretrain_ckpt_path=pretrained_optical_flow_path)

        blocks = []
        for _ in range(layers):
            blocks.append(TransformerBlock(heads = heads, hidden = self.channels))

        self.transformer = M.Sequential(*blocks)

        self.encoder = M.Sequential(
            M.Conv2d(3, 64, 3, 2, 1),
            M.LeakyReLU(0.2),
            M.Conv2d(64, 64, 3, 1, 1),
            M.LeakyReLU(0.2),
            M.Conv2d(64, 128, 3, 2, 1),
            M.LeakyReLU(0.2),
            M.Conv2d(128, self.channels, 3, 1, 1),
            M.LeakyReLU(0.2)
        )

        self.decoder = M.Sequential(
            deconv(2*self.channels, 128, kernel_size=3, padding=1),
            M.LeakyReLU(0.2),
            M.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            M.LeakyReLU(0.2),
            deconv(64, 64, kernel_size=3, padding=1),
            M.LeakyReLU(0.2)
        )
        self.lastconv = M.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        self.init_weights()

    def forward(self, masked_frames, masks):
        b, t, c, h, w = masked_frames.shape
        masks = masks.reshape(b*t, 1, h, w)
        enc_feat = self.encoder(masked_frames.reshape(b*t, c, h, w))
        _, c, h, w = enc_feat.shape
        masks = F.nn.interpolate(masks, scale_factor = 1.0/4, mode='BILINEAR', align_corners=False)
        enc_feat = self.transformer({"x":enc_feat, "m":masks, "b":b, "c": c})['x']
        enc_feat = enc_feat.reshape(b, t, c, h, w)
        # for in t dim
        output = [] # t-1 * (b,2,h,w)
        for i in range(t-1):
            pre = enc_feat[:, i, :, :, :]
            next = enc_feat[:, i+1, :, :, :]
            output.append(self.lastconv(self.decoder(F.concat([pre, next], axis=1))))
        output = F.stack(output, axis=1) # (b,t-1,2,h,w)
        output = output.reshape(b*(t-1), 2, 4*h, 4*w)
        return output

    def init_weights(self, pretrained = False):
        if pretrained:
            path = "./workdirs/sttn.mge"
            print("loading pretrained model for G 🤡🤡🤡🤡🤡🤡...")
            state_dict = megengine.load(path)
            self.load_state_dict(state_dict, strict=True)
        else:
            default_init_weights(self.encoder, scale=1, nonlinearity="leaky_relu", lrelu_value = 0.2)
            default_init_weights(self.decoder, scale=1, nonlinearity="leaky_relu", lrelu_value = 0.2)

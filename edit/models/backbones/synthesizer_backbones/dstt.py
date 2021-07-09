"""
    DSTT
    do not use mask attention
"""
import numpy as np
import megengine
import megengine.module as M
from megengine import Parameter
import megengine.functional as F
from megengine.module.init import ones_, zeros_
from edit.models.builder import BACKBONES
from edit.models.common import default_init_weights, PixelShufflePack
import math

class LayerNorm3d(M.Module):
    def __init__(self, num_channels, eps=1e-05, affine=True):
        super(LayerNorm3d, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(np.ones(num_channels, dtype="float32"))
            self.bias = Parameter(np.zeros(num_channels, dtype="float32"))
        else:
            self.weight = None
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            ones_(self.weight)
            zeros_(self.bias)

    def forward(self, x):
        N, n, C = x.shape
        assert C == self.num_channels
        x = x.reshape(N*n, -1)
        # NOTE mean will keepdims in next two lines.
        mean = x.mean(axis=1, keepdims=1)
        var = (x ** 2).mean(axis=1, keepdims=1) - mean * mean

        x = (x - mean) / F.sqrt(var + self.eps)
        x = x.reshape(N, n, C)
        if self.affine:
            x = self.weight.reshape(1, 1, -1) * x + self.bias.reshape(1, 1, -1)

        return x

    def _module_info_string(self) -> str:
        s = "channels={num_channels}, eps={eps}, affine={affine}"
        return s.format(**self.__dict__)

class HierarchyEncoder(M.Module):
    def __init__(self, channel):
        super(HierarchyEncoder, self).__init__()
        assert channel == 256
        self.group = [1, 2, 4, 8, 1]
        self.layers = [
            M.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1),
            M.LeakyReLU(0.2),
            M.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, groups=2),
            M.LeakyReLU(0.2),
            M.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, groups=4),
            M.LeakyReLU(0.2),
            M.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, groups=8),
            M.LeakyReLU(0.2),
            M.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, groups=1),
            M.LeakyReLU(0.2)
        ]

    def forward(self, x):
        bt, c, h, w = x.shape
        out = x
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and i != 0:
                g = self.group[i//2]
                x0 = x.reshape(bt, g, -1, h, w)
                out0 = out.reshape(bt, g, -1, h, w)
                out = F.concat([x0, out0], 2).reshape(bt, -1, h, w)
            out = layer(out)
        return out

class FeedForward(M.Module):
    def __init__(self, d_model, p=0.1):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.conv = M.Sequential(
            M.Linear(d_model, d_model * 4),
            M.ReLU(),
            M.Dropout(p),
            M.Linear(d_model * 4, d_model),
            M.Dropout(p)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Attention(M.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def __init__(self, p=0.1):
        super(Attention, self).__init__()
        self.dropout = M.Dropout(p)

    def forward(self, query, key, value):
        scores = F.matmul(query, key.transpose(0, 1, 3, 2)) / math.sqrt(query.shape[-1]) # b, t*out_h*out_w, t*out_h*out_w
        p_attn = F.nn.softmax(scores, axis=3)
        p_attn = self.dropout(p_attn)
        p_val = F.matmul(p_attn, value)
        return p_val, p_attn

class MultiHeadedAttention(M.Module):
    def __init__(self, d_model, head, mode, p=0.1):
        super().__init__()
        self.mode = mode
        self.query_embedding = M.Linear(d_model, d_model)
        self.key_embedding = M.Linear(d_model, d_model)
        self.value_embedding = M.Linear(d_model, d_model)
        self.output_linear = M.Linear(d_model, d_model)
        self.attention = Attention(p=p)
        self.head = head

    def do_linear(self, x, model):
        bt, hw, c = x.shape
        out = x.reshape(-1, c)
        out = model(out)
        return out.reshape(bt, hw, c)        

    def forward(self, x, t, h, w):
        bt, n, c = x.shape
        assert n == h * w
        b = bt // t
        assert b == 1
        c_h = c // self.head
        query = self.do_linear(x, self.query_embedding)
        key = self.do_linear(x, self.key_embedding)
        value = self.do_linear(x, self.value_embedding)
        if self.mode == 's':
            query = query.reshape(bt, n, self.head, c_h).transpose(0, 2, 1, 3)
            key = key.reshape(bt, n, self.head, c_h).transpose(0, 2, 1, 3) # [bt, head, n, c_h]
            value = value.reshape(bt, n, self.head, c_h).transpose(0, 2, 1, 3)
            att, _ = self.attention(query, key, value)
            att = att.transpose(0, 2, 1, 3).reshape(bt, n, c)
        elif self.mode == 't':
            key = key.reshape(t, 2, h//2, 2, w//2, self.head, c_h)
            key = key.transpose(1, 3, 5, 0, 2, 4, 6).reshape(b*4, self.head, -1, c_h)
            query = query.reshape(t, 2, h//2, 2, w//2, self.head, c_h)
            query = query.transpose(1, 3, 5, 0, 2, 4, 6).reshape(b*4, self.head, -1, c_h)
            value = value.reshape(t, 2, h//2, 2, w//2, self.head, c_h)
            value = value.transpose(1, 3, 5, 0, 2, 4, 6).reshape(b*4, self.head, -1, c_h)
            att, _ = self.attention(query, key, value)
            att = att.reshape(2, 2, self.head, t, h//2, w//2, c_h)
            att = att.transpose(3, 0, 4, 1, 5, 2, 6).reshape(bt, n, c)
        output = self.do_linear(att, self.output_linear)
        return output

class TransformerBlock(M.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """
    def __init__(self, hidden, num_head=4, mode='s', dropout=0.1):
        super().__init__()
        self.attention = MultiHeadedAttention(d_model=hidden, head=num_head, mode=mode, p=dropout)
        self.ffn = FeedForward(hidden, p=dropout)
        self.norm1 = LayerNorm3d(hidden)
        self.norm2 = LayerNorm3d(hidden)
        self.dropout = M.Dropout(dropout)
        
    def forward(self, input):
        x, t = input['x'], input['t']
        h, w = input['h'], input['w']
        x = self.norm1(x)
        x = x + self.dropout(self.attention(x, t, h, w))
        y = self.norm2(x)
        x = x + self.ffn(y)
        return {'x': x, 't': t, 'h':h, 'w':w}

class deconv(M.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super(deconv, self).__init__()
        self.conv = M.Conv2d(input_channel, output_channel, kernel_size, 1, padding)

    def forward(self, x):
        x = F.nn.interpolate(x, scale_factor = 2, mode='BILINEAR', align_corners=True)
        return self.conv(x)

@BACKBONES.register_module()
class DSTT(M.Module):
    def __init__(self,
                 layers = 8,
                 heads = 4,
                 dropout = 0.):
        super(DSTT, self).__init__()
        assert layers % 2 == 0
        channels = 256
        hidden = 512
        kernel_size = (5, 5)
        padding = (2, 2)
        stride  = (2, 2)
        blocks = []
        for _ in range(layers // 2):
            blocks.append(TransformerBlock(hidden=hidden, num_head=heads, mode='t', dropout=dropout))
            blocks.append(TransformerBlock(hidden=hidden, num_head=heads, mode='s', dropout=dropout))

        self.transformer = M.Sequential(*blocks)
        self.patch2vec = M.Conv2d(channels//2, hidden, kernel_size=kernel_size, stride=stride, padding=padding)
        self.vec2patch = PixelShufflePack(hidden, channels//2, scale_factor=2, upsample_kernel=3)

        self.encoder = M.Sequential(
            M.Conv2d(3, 64, 3, 2, 1),
            M.LeakyReLU(0.2),
            M.Conv2d(64, 64, 3, 1, 1),
            M.LeakyReLU(0.2),
            M.Conv2d(64, 128, 3, 2, 1),
            M.LeakyReLU(0.2),
            M.Conv2d(128, channels, 3, 1, 1),
            M.LeakyReLU(0.2)
        )
        self.hier_enc = HierarchyEncoder(channels) # 256 -> 128
        self.decoder = M.Sequential(
            deconv(channels//2, 128, kernel_size=3, padding=1),
            M.LeakyReLU(0.2),
            M.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            M.LeakyReLU(0.2),
            deconv(64, 64, kernel_size=3, padding=1),
            M.LeakyReLU(0.2)
        )
        self.lastconv = M.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.init_weights()

    def forward(self, masked_frames, masks):
        b, t, c, h, w = masked_frames.shape
        masked_frames = F.vision.interpolate(masked_frames.reshape(b*t, c, h, w), scale_factor=0.5, align_corners=True)
        _, _, h, w = masked_frames.shape
        enc_feat = self.encoder(masked_frames)
        enc_feat = self.hier_enc(enc_feat)
        trans_feat = self.patch2vec(enc_feat)
        _, c, h, w = trans_feat.shape
        trans_feat = trans_feat.reshape(b*t, c, -1).transpose(0, 2, 1)
        trans_feat = self.transformer({"x":trans_feat, "t": t, 'h': h, 'w':w})['x']
        trans_feat = trans_feat.transpose(0, 2, 1).reshape(b*t, c, h, w)
        trans_feat = self.vec2patch(trans_feat)
        enc_feat = enc_feat + trans_feat

        output = self.decoder(enc_feat)
        output = self.lastconv(output)
        output = F.tanh(output)
        output = F.nn.interpolate(output, scale_factor = 2, mode='BILINEAR', align_corners=True)
        return output

    def init_weights(self, pretrained = False):
        if pretrained:
            path = "./workdirs/dstt.mge"
            print("loading pretrained model for G 🤡🤡🤡🤡🤡🤡...")
            state_dict = megengine.load(path)
            self.load_state_dict(state_dict, strict=True)
        else:
            default_init_weights(self.encoder, scale=1, nonlinearity="leaky_relu", lrelu_value = 0.2)
            default_init_weights(self.hier_enc, scale=1, nonlinearity="leaky_relu", lrelu_value = 0.2)
            default_init_weights(self.decoder, scale=1, nonlinearity="leaky_relu", lrelu_value = 0.2)
            default_init_weights(self.lastconv, scale=1, nonlinearity="tanh")

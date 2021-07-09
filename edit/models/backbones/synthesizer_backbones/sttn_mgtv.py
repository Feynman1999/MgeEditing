"""
    用于mgtv的sttn (baseline)
    区别：attention时, blocksize不同
"""
import numpy as np
import megengine
import megengine.module as M
import megengine.functional as F
from edit.models.builder import BACKBONES
from edit.models.common import default_init_weights
import math

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
class STTN_MGTV(M.Module):
    def __init__(self,
                 channels = 64,
                 layers = 8,
                 heads = 4):
        super(STTN_MGTV, self).__init__()

        self.channels = channels * heads

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
            deconv(self.channels, 128, kernel_size=3, padding=1),
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
        masks = masks.reshape(b*t, 1, h, w)
        enc_feat = self.encoder(masked_frames.reshape(b*t, c, h, w))
        _, c, h, w = enc_feat.shape
        masks = F.nn.interpolate(masks, scale_factor = 1.0/4, mode='BILINEAR', align_corners=False)
        enc_feat = self.transformer({"x":enc_feat, "m":masks, "b":b, "c": c})['x']
        output = self.decoder(enc_feat)
        output = self.lastconv(output)
        output = F.tanh(output)
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
            default_init_weights(self.lastconv, scale=1, nonlinearity="tanh")
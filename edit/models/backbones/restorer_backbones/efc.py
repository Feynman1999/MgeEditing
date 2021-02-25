import numpy as np
import megengine
import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
from edit.models.builder import BACKBONES
from edit.models.common import add_H_W_Padding
import math

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
        output = output.transpose(0, 1, 4, 3, 5, 2)
        # N C oH oW
        output = output.reshape(N, oC, oH, oW)
        return output

class FeedForward(M.Module):
    def __init__(self, d_model, layer_norm = False):
        super(FeedForward, self).__init__()
        if layer_norm:
            self.conv = M.Sequential(
                M.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2, stride=1),
                M.normalization.LayerNorm(d_model),
                M.ReLU(),
                M.Conv2d(d_model, d_model, kernel_size=3, padding=1, dilation=1, stride=1),
            )
        else:
            self.conv = M.Sequential(
                M.ConvRelu2d(d_model, d_model, kernel_size=3, padding=2, dilation=2, stride=1),
                M.Conv2d(d_model, d_model, kernel_size=3, padding=1, dilation=1, stride=1),
            )

    def forward(self, x):
        return x + self.conv(x)
        
def do_attention(query, key, value):
    # print(query.shape, key.shape)
    scores = F.matmul(query, key.transpose(0, 2, 1)) / math.sqrt(query.shape[-1]) # b, t*out_h*out_w, t*out_h*out_w
    # print(scores.shape)
    p_attn = F.nn.softmax(scores, axis=2)
    # print(p_attn.shape, value.shape)
    p_val = F.matmul(p_attn, value)
    return p_val, p_attn

class MultiHeadedAttention(M.Module):
    def __init__(self, heads, hidden, layer_norm):
        super(MultiHeadedAttention, self).__init__()
        self.headnums = heads
        self.query_embedding = M.Conv2d(hidden, hidden, kernel_size=1, padding=0)
        self.key_embedding = M.Conv2d(hidden, hidden, kernel_size=1, padding=0)
        self.value_embedding = M.Conv2d(hidden, hidden, kernel_size=1, padding=0)
        
        if layer_norm:
            self.output_linear = M.Sequential(
                                    M.Conv2d(hidden, hidden, kernel_size=3, padding=1, stride=1),
                                    M.normalization.LayerNorm(hidden),
                                    M.ReLU()
                                )
        else:
            self.output_linear = M.ConvRelu2d(hidden, hidden, kernel_size=3, padding=1, stride=1)

    def forward(self, frames, trans_flags):
        t = len(frames)
        b,c,_,_ = frames[0].shape
        assert b == 1
        h_list = [ item.shape[-2] for item in frames]
        w_list = [ item.shape[-1] for item in frames]
        # print(h_list, w_list)
        d_k = c // self.headnums  # 每个head的通道数
        outputs = []
        for _ in range(t):
            outputs.append([])
        _query = [self.query_embedding(item) for item in frames]
        _key = [self.key_embedding(item) for item in frames]
        _value = [self.value_embedding(item) for item in frames]
        for idx in range(self.headnums):
            height, width = 9, 8
            if trans_flags[0]:
                height, width = width, height
            query = [ item[:, (idx*d_k):(idx*d_k + d_k), ...] for item in _query ]
            key = [ item[:, (idx*d_k):(idx*d_k + d_k), ...] for item in _key ]
            value = [ item[:, (idx*d_k):(idx*d_k + d_k), ...] for item in _value ]
            # 1) embedding and reshape
            # for every frame, (1, C, h, w) -> (out_h * out_w, d_k * height * width)
            record_split_array = [] # record split array for every frame
            query_list = []
            for i, item in enumerate(query):
                now_item = item.reshape(1, d_k, h_list[i]//height, height, w_list[i]//width, width)
                now_item = now_item.transpose(0, 2, 4, 1, 3, 5).reshape(1, -1, d_k*height*width)
                record_split_array.append(now_item.shape[1])
                query_list.append(now_item)
            query = F.concat(query_list, axis = 1)
            del query_list
            key_list = []
            for i, item in enumerate(key):
                now_item = item.reshape(1, d_k, h_list[i]//height, height, w_list[i]//width, width)
                now_item = now_item.transpose(0, 2, 4, 1, 3, 5).reshape(1, -1, d_k*height*width)
                key_list.append(now_item)
            key = F.concat(key_list, axis = 1)
            del key_list
            value_list = []
            for i, item in enumerate(value):
                now_item = item.reshape(1, d_k, h_list[i]//height, height, w_list[i]//width, width)
                now_item = now_item.transpose(0, 2, 4, 1, 3, 5).reshape(1, -1, d_k*height*width)
                value_list.append(now_item)
            value = F.concat(value_list, axis = 1)
            del value_list        
            # print(record_split_array)
            # 2) Apply attention on all the projected vectors in batch.
            y, _ = do_attention(query, key, value)
            ans_y = []
            now_deal = 0
            for item in record_split_array:
                ans_y.append(y[:, now_deal:now_deal + item, :])
                now_deal += item
            del y
            # 将ans_y 中的tensor reshape
            for i, item in enumerate(ans_y):
                now_item = item.reshape(1, h_list[i]//height, w_list[i]//width, d_k, height, width)
                now_item = now_item.transpose(0, 3, 1, 4, 2, 5).reshape(1, d_k, h_list[i], w_list[i])
                outputs[i].append(now_item)
        for i in range(t):
            outputs[i] = F.concat(outputs[i], axis=1)
            
        return [self.output_linear(item) for item in outputs]

class TransformerBlock(M.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """
    def __init__(self, heads, hidden, layer_norm):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads, hidden, layer_norm)
        self.feed_forward = FeedForward(hidden, layer_norm)

    def forward(self, Dict):
        # frames: list of tensor [1,c,h,w]
        frames = Dict['x']
        trans_flags = Dict['trans']
        attentioned_frames = self.attention(frames, trans_flags)
        for i in range(len(attentioned_frames)):
            attentioned_frames[i] = attentioned_frames[i] + frames[i]
        return {'x': [ self.feed_forward(item) for item in attentioned_frames], 'trans':trans_flags}


@BACKBONES.register_module()
class EFC(M.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 channels = 32,
                 layers = 6,
                 heads = 4,
                 upscale_factor=4,
                 layer_norm = True):
        super(EFC, self).__init__()
        self.upscale_factor = upscale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_norm = layer_norm

        self.channels = channels * heads

        blocks = []
        for _ in range(layers):
            blocks.append(TransformerBlock(heads = heads, hidden = self.channels, layer_norm = layer_norm))

        self.transformer = M.Sequential(*blocks)

        self.encoder = M.Sequential(
            M.Conv2d(self.in_channels, 48, kernel_size = 3, stride=1, padding=1),
            M.LeakyReLU(0.2),
            FeedForward(48, layer_norm=False),
            M.Conv2d(48, self.channels, kernel_size=3, stride=1, padding=1),
            M.LeakyReLU(0.2),
            FeedForward(self.channels, layer_norm=False),
            FeedForward(self.channels, layer_norm=False)
        )

        self.decoder = M.Sequential(
            M.ConvRelu2d(self.channels, 256, kernel_size=3, stride=1, padding=1),
            M.ConvRelu2d(256, 512, kernel_size=3, stride=1, padding=1),
            PixelShuffle(scale=2),
            M.ConvRelu2d(128, 256, kernel_size=3, stride=1, padding=1),
            PixelShuffle(scale=2),
            M.ConvRelu2d(64, 64, kernel_size=3, stride=1, padding=1),
            M.Conv2d(64, self.out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, frames, trans_flags):
        # 对frames做encoder处理
        enc_feat = [ self.encoder(item) for item in frames]
        enc_feat = self.transformer({'x':enc_feat, 'trans':trans_flags})['x'] # list of tensor
        enc_feat = [ self.decoder(item) for item in enc_feat]
        return enc_feat

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

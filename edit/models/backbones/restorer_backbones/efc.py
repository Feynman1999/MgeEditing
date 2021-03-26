import numpy as np
import megengine
import megengine.module as M
from megengine.module.conv import Conv2d
import megengine.functional as F
from edit.models.builder import BACKBONES
from edit.models.common import add_H_W_Padding, default_init_weights, ResBlock, ResBlocks, PixelShufflePack
import math

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
            self.conv = ResBlock(d_model, d_model, init_scale=0.1)

    def forward(self, x):
        return self.conv(x)

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
        height, width = 9, 8
        if trans_flags[0]:
            height, width = width, height

        t = len(frames)
        b,c,_,_ = frames[0].shape
        assert b == 1
        h_list = [ item.shape[-2] for item in frames]
        w_list = [ item.shape[-1] for item in frames]
        # print(h_list, w_list)
        d_k = c // self.headnums  # 每个head的通道数

        _query = [self.query_embedding(item) for item in frames]
        _key = [self.key_embedding(item) for item in frames]
        _value = [self.value_embedding(item) for item in frames]
        
        outputs = []

        # 将每一帧的feature reshape到batch上，增加并行度（分块大小一样）
        query = [ item.reshape(b*self.headnums, d_k, item.shape[-2], item.shape[-1]) for item in _query ]
        key = [ item.reshape(b*self.headnums, d_k, item.shape[-2], item.shape[-1]) for item in _key ]
        value = [ item.reshape(b*self.headnums, d_k, item.shape[-2], item.shape[-1]) for item in _value ]
        record_split_array = [] # record split array for every frame
        query_list = []
        for i, item in enumerate(query):
            now_item = item.reshape(b*self.headnums, d_k, h_list[i]//height, height, w_list[i]//width, width)
            now_item = now_item.transpose(0, 2, 4, 1, 3, 5).reshape(b*self.headnums, -1, d_k*height*width)
            record_split_array.append(now_item.shape[1])
            query_list.append(now_item)
        query = F.concat(query_list, axis = 1)
        del query_list
        key_list = []
        for i, item in enumerate(key):
            now_item = item.reshape(b*self.headnums, d_k, h_list[i]//height, height, w_list[i]//width, width)
            now_item = now_item.transpose(0, 2, 4, 1, 3, 5).reshape(b*self.headnums, -1, d_k*height*width)
            key_list.append(now_item)
        key = F.concat(key_list, axis = 1)
        del key_list
        value_list = []
        for i, item in enumerate(value):
            now_item = item.reshape(b*self.headnums, d_k, h_list[i]//height, height, w_list[i]//width, width)
            now_item = now_item.transpose(0, 2, 4, 1, 3, 5).reshape(b*self.headnums, -1, d_k*height*width)
            value_list.append(now_item)
        value = F.concat(value_list, axis = 1)
        del value_list
        y, _ = do_attention(query, key, value)
        ans_y = []
        now_deal = 0
        for item in record_split_array:
            ans_y.append(y[:, now_deal:now_deal + item, :])
            now_deal += item
        del y
        # 将ans_y 中的tensor reshape, 一个tensor表示一张图片
        for i, item in enumerate(ans_y):
            now_item = item.reshape(b*self.headnums, h_list[i]//height, w_list[i]//width, d_k, height, width)
            now_item = now_item.transpose(0, 3, 1, 4, 2, 5).reshape(b, c, h_list[i], w_list[i])
            outputs.append(now_item)

        # outputs = []
        # for _ in range(t):
        #     outputs.append([])
        # for idx in range(self.headnums):
        #     height, width = 9, 8
        #     if trans_flags[0]:
        #         height, width = width, height
        #     query = [ item[:, (idx*d_k):(idx*d_k + d_k), ...] for item in _query ]
        #     key = [ item[:, (idx*d_k):(idx*d_k + d_k), ...] for item in _key ]
        #     value = [ item[:, (idx*d_k):(idx*d_k + d_k), ...] for item in _value ]
        #     # 1) embedding and reshape
        #     # for every frame, (1, C, h, w) -> (out_h * out_w, d_k * height * width)
        #     record_split_array = [] # record split array for every frame
        #     query_list = []
        #     for i, item in enumerate(query):
        #         now_item = item.reshape(1, d_k, h_list[i]//height, height, w_list[i]//width, width)
        #         now_item = now_item.transpose(0, 2, 4, 1, 3, 5).reshape(1, -1, d_k*height*width)
        #         record_split_array.append(now_item.shape[1])
        #         query_list.append(now_item)
        #     query = F.concat(query_list, axis = 1)
        #     del query_list
        #     key_list = []
        #     for i, item in enumerate(key):
        #         now_item = item.reshape(1, d_k, h_list[i]//height, height, w_list[i]//width, width)
        #         now_item = now_item.transpose(0, 2, 4, 1, 3, 5).reshape(1, -1, d_k*height*width)
        #         key_list.append(now_item)
        #     key = F.concat(key_list, axis = 1)
        #     del key_list
        #     value_list = []
        #     for i, item in enumerate(value):
        #         now_item = item.reshape(1, d_k, h_list[i]//height, height, w_list[i]//width, width)
        #         now_item = now_item.transpose(0, 2, 4, 1, 3, 5).reshape(1, -1, d_k*height*width)
        #         value_list.append(now_item)
        #     value = F.concat(value_list, axis = 1)
        #     del value_list        
        #     # print(record_split_array)
        #     # 2) Apply attention on all the projected vectors in batch.
        #     y, _ = do_attention(query, key, value)
        #     ans_y = []
        #     now_deal = 0
        #     for item in record_split_array:
        #         ans_y.append(y[:, now_deal:now_deal + item, :])
        #         now_deal += item
        #     del y
        #     # 将ans_y 中的tensor reshape
        #     for i, item in enumerate(ans_y):
        #         now_item = item.reshape(1, h_list[i]//height, w_list[i]//width, d_k, height, width)
        #         now_item = now_item.transpose(0, 3, 1, 4, 2, 5).reshape(1, d_k, h_list[i], w_list[i])
        #         outputs[i].append(now_item)
        # for i in range(t):
        #     outputs[i] = F.concat(outputs[i], axis=1)
            
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
        
        self.conv1 = M.Conv2d(self.in_channels, self.channels, kernel_size = 3, stride=1, padding=1)
        self.encoder = M.Sequential(
            FeedForward(self.channels, layer_norm=layer_norm),
            FeedForward(self.channels, layer_norm=layer_norm),
            FeedForward(self.channels, layer_norm=layer_norm)
        )
        
        self.reconstruction = ResBlocks(channel_num=self.channels, resblock_num=3)    
        self.upsample1 = PixelShufflePack(self.channels, self.channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(self.channels, 64, 2, upsample_kernel=3)
        self.conv_hr = M.Conv2d(64, 64, 3, 1, 1)  # need init
        self.conv_last = M.Conv2d(64, out_channels, 3, 1, 1)
        self.lrelu = M.LeakyReLU(negative_slope=0.05)
    
    def do_upsample(self, x):
        out = self.reconstruction(x)
        out = self.lrelu(self.upsample1(out))
        out = self.lrelu(self.upsample2(out))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        return out

    def forward(self, frames, trans_flags):
        # 对frames做encoder处理
        enc_feat = [ self.encoder(self.conv1(item)) for item in frames]
        enc_feat = self.transformer({'x':enc_feat, 'trans':trans_flags})['x'] # list of tensor
        enc_feat = [ self.do_upsample(item) for item in enc_feat]
        return enc_feat

    def init_weights(self, pretrained=None):
        default_init_weights(self.conv1)
        default_init_weights(self.conv_hr, nonlinearity='leaky_relu')

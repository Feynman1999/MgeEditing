import numpy as np
import megengine
import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
from edit.models.builder import BACKBONES
from edit.models.restorers.MBR.utils import cal_flow_mask
from edit.models.common import add_H_W_Padding, default_init_weights, ResBlock, ResBlocks, PixelShufflePack
from .basicVSR_multi_layer import Spynet, backwarp
import math
import time

test_t_thr = 21

class Basic(M.Module):
    def __init__(self, channels, out_c):
        super(Basic, self).__init__()
        self.netBasic = M.Sequential(
            Conv2d(in_channels=channels + out_c, out_channels=64, kernel_size=1, stride=1, padding=0),
            M.ReLU(),
            Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2),
            M.ReLU(),
            Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            M.ReLU(),
            Conv2d(in_channels=32, out_channels=out_c, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, tenInput):
        return self.netBasic(tenInput)

class Spynet2(M.Module):
    """
        每个head attention的位置不同

        offsets_num: 对于每个head，学习的offset的数量，默认为2
        每个块attention的数量为(keept*2+1) * (1 + offsets_num)
    """
    def __init__(self, channels, offsets_num = 2):
        super(Spynet2, self).__init__()
        self.num_layers = 3
        self.heads = 4 # for transformer
        self.offsets_num = offsets_num
        self.out_c = self.heads * self.offsets_num * 2
        assert offsets_num in (1,2,3,4)

        basic_list = [ Basic(channels, self.out_c) for _ in range(self.num_layers) ]
        self.netBasic = M.Sequential(*basic_list)
        self.init_weights()

    def cal_resize_value(self, x, resize_multi = 32):
        diff = x % resize_multi
        if diff == 0:
            return x
        return x + resize_multi - diff

    def forward(self, ten):
        # 如果输入的图片不是8的倍数则进行resize
        _,_,H,W = ten.shape
        resize_flag = False
        if H % 8 != 0 or W % 8 != 0:
            resize_flag = True
            # 算出需要resize到的大小
            aim_H = self.cal_resize_value(H, resize_multi=8)
            aim_W = self.cal_resize_value(W, resize_multi=8)
            ten = F.nn.interpolate(ten, size=[aim_H, aim_W], align_corners=False)

        ten = [ten]
        for intLevel in range(self.num_layers - 1):
            ten.insert(0, F.avg_pool2d(inp=ten[0], kernel_size=2, stride=2))

        tenFlow = F.zeros([ten[0].shape[0], self.out_c, ten[0].shape[2], ten[0].shape[3]])
        tenUpsampled = tenFlow
        tenFlow = self.netBasic[0]( F.concat([ten[0], tenUpsampled], axis=1) )
        for intLevel in range(1, len(ten)):
            tenUpsampled = F.nn.interpolate(inp=tenFlow, scale_factor=2, mode='BILINEAR', align_corners=True) * 2.0
            tenFlow = self.netBasic[intLevel]( F.concat([ten[intLevel], tenUpsampled], axis=1) ) + tenUpsampled
        
        # adjust the flow values
        if resize_flag:
            tenFlow = F.nn.interpolate(inp=tenFlow, size=[H, W], align_corners=False)
            for i in range(0, self.out_c, 2):
                tenFlow[:, i, :, :] *= float(W) / float(aim_W)
                tenFlow[:, i+1, :, :] *= float(H) / float(aim_H)
        # tenFlow: [b, heads * num * 2, h, w]
        return tenFlow.reshape(-1, self.heads, self.offsets_num, 2, H, W)

    def init_weights(self):
        default_init_weights(self.netBasic, scale=0.1)

class FeedForward(M.Module):
    def __init__(self, d_model, layer_norm = False):
        super(FeedForward, self).__init__()
        self.layer_norm = layer_norm
        if layer_norm:
            self.conv = M.Sequential(
                M.Conv2d(d_model, d_model, kernel_size=3, padding=1, stride=1),
                M.normalization.LayerNorm(d_model),
                M.ReLU(),
                M.Conv2d(d_model, d_model, kernel_size=3, padding=1, stride=1)
            )
        else:
            self.conv = ResBlock(d_model, d_model, init_scale=0.1)

    def forward(self, x):
        if self.layer_norm:
            return x + self.conv(x)
        else:
            return self.conv(x)

def do_attention(query, key, value):
    scores = F.matmul(query, key.transpose(0, 2, 1)) / math.sqrt(query.shape[-1]) # b, t*out_h*out_w, t*out_h*out_w
    p_attn = F.nn.softmax(scores, axis=2)
    p_val = F.matmul(p_attn, value)
    return p_val, p_attn

class MultiHeadedAttention(M.Module):
    def __init__(self, heads, hidden, layer_norm, layer_id, keept, offsets_num):
        super(MultiHeadedAttention, self).__init__()
        self.layer_id = layer_id # start from 0
        self.headnums = heads
        assert heads % 4 == 0
        self.qk_reduction = 1
        if heads > 0:
            assert hidden % (self.qk_reduction*heads) == 0
        self.blocks = [(9, 8), (15, 8), (15, 16), (15, 20)]
        self.keept = keept # can be 1,2,3
        assert keept in (1, 2, 3)
        self.offsets_num = offsets_num
        self.layer_norm = layer_norm

        if heads>0:
            self.query_embedding = M.Conv2d(hidden, hidden // self.qk_reduction, kernel_size=1, padding=0)
            self.key_embedding = M.Conv2d(hidden, hidden // self.qk_reduction, kernel_size=1, padding=0) # for save memory
            self.value_embedding = M.Conv2d(hidden, hidden, kernel_size=1, padding=0)
        else:
            self.reduction = M.ConvRelu2d((keept *2+1)*hidden, hidden, kernel_size=3, padding=1)

        if layer_norm:
            self.output_linear = M.Sequential(
                                    M.Conv2d(hidden, hidden, kernel_size=3, padding=1, stride=1),
                                    M.normalization.LayerNorm(hidden),
                                    M.ReLU()
                                )
        else:
            self.output_linear = M.Sequential(
                M.Conv2d(hidden, hidden, kernel_size=3, padding=1, stride=1),
                M.LeakyReLU(negative_slope=0.1)
            )

        self.init_weights()

    def do_q_k_v_outputlinear(self, feat, t, model):
        if t > test_t_thr: # for test
            assert feat.shape[0] == 1*t and len(feat.shape) == 4
            res = []
            for i in range(t):
                res.append(model(feat[i:(i+1), ...]))
            return F.concat(res, axis=0)
        else:
            return model(feat)

    def warp_given_offset(self, x, offset, head_idx):
        """
            x: [bt, (2*keept+1)*d_k, h, w]
            offset: [bt, offsetnum, 2, h, w] 
            return : [bt, (2*keept+1)*(offsetnum+1), d_k, h, w]
        """
        offsetnum = offset.shape[1]
        concat_feats = [x]
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (0, -1), (1, -1), (1, 0), (1, 1)]
        for idx in range(offsetnum):
            # print("idx: {} value: {}".format(idx, offset[:, idx, ...].mean()))
            shift_pixels = 4
            dir_idx = (head_idx * offsetnum + idx) % len(directions)
            shift_x = directions[dir_idx][0] * shift_pixels
            shift_y = directions[dir_idx][1] * shift_pixels
            flow = F.concat([offset[:, idx, 0:1, ...] + shift_x, offset[:, idx, 1:2, ...] + shift_y], axis=1)
            concat_feats.append(backwarp(x, flow, border_mode="REPLICATE"))
        return F.concat(concat_feats, axis=1)

    def warp_given_list(self, L, flow, now_frame, dim, mask = None):
        need_warp = L[-1]
        B, C, _, _ = need_warp.shape
        assert C % dim == 0
        if C // dim > self.keept:
            need_warp = need_warp[:, dim: , ...] # 去除一个 
        warped_key = backwarp(need_warp, flow, border_mode="REPLICATE")
        # 对warped_key进行mask操作
        if mask is not None:
            B, C, _, _ = warped_key.shape
            # 每dim分别做
            for i in range(C // dim):
                warped_key[:, i*dim:(i+1)*dim, ...] *= mask
                warped_key[:, i*dim:(i+1)*dim, ...] += (1-mask) * now_frame
        L.append(F.concat([warped_key, now_frame], axis=1))

    def time_attention(self, _query, _key, _value, offsets = None):
        bt, qkc, h, w = _query.shape
        _, _, vc, h, w = _value.shape
        d_qk = qkc // self.headnums
        d_v = vc // self.headnums
        assert qkc % self.headnums ==0 and vc % self.headnums == 0
        outputs = [] # 每个head的结果
        for idx in range(self.headnums):
            height, width = self.blocks[idx % 4]
            out_h, out_w = h // height, w // width
            query = _query[:, (idx*d_qk):(idx*d_qk + d_qk), ...]
            query = query.reshape(bt, d_qk, out_h, height, out_w, width)
            query = query.transpose(0, 2, 4, 1, 3, 5).reshape((bt*out_h*out_w, 1, d_qk*height*width))

            key = _key[:, :, (idx*d_qk):(idx*d_qk + d_qk), ...].reshape(bt, -1, h, w) 
            value = _value[:, :, (idx*d_v):(idx*d_v + d_v), ...].reshape(bt, -1, h, w)

            if offsets is not None:
                key = self.warp_given_offset(key, offsets[:, idx, ...], idx)
                value = self.warp_given_offset(value, offsets[:, idx, ...], idx)

            key = key.reshape(bt, -1, d_qk, out_h, height, out_w, width)
            key = key.transpose(0, 3, 5, 1, 2, 4, 6).reshape((bt*out_h*out_w, -1, d_qk*height*width))
            value = value.reshape(bt, -1, d_v, out_h, height, out_w, width)
            value = value.transpose(0, 3, 5, 1, 2, 4, 6).reshape((bt*out_h*out_w, -1, d_v*height*width))
            y, _ = do_attention(query, key, value)
            y = y.reshape(bt, out_h, out_w, d_v, height, width)
            y = y.transpose((0, 3, 1, 4, 2, 5)).reshape(bt, d_v, h, w)
            outputs.append(y)
        outputs = F.concat(outputs, axis = 1)
        return outputs

    def test_time_attention(self, _query, key, value, t, b, flows_0_1, flows_1_0, offsets, mask_0_1 = None, mask_1_0 = None):
        """
            循环每一帧，分别进行处理
            如果是中间帧，如keept=2时，下标为2,3,4...97的帧，设为i，直接取 i-2, i-1, i, i+1, i+2，以及对应的flow和offset即可
            （实际上训练时的顺序为i-2,i-1,i,i+2,i+1）但无所谓，因为时间上没有postional encoding
        """
        assert b == 1, "for test, make sure batchsize = 1"
        bt, c, h, w = _query.shape
        assert bt == t
        outputs = [] # 每一帧的结果
        """
            遍历每一帧，得到必要的key和value
            每一次搞一个list，把需要的依次加入
        """
        for i in range(t):
            # 确定left和right
            if i < self.keept:
                left = 0
                right = i + self.keept
            elif i >= t - self.keept:
                left = i - self.keept
                right = t - 1
            else:
                left = i - self.keept
                right = i + self.keept

            # 前面的warp到当前帧
            now_frame_key_f = key[left:left+1, ...]
            now_frame_value_f = value[left:left+1, ...]
            for j in range(left+1, i+1):
                now_flow = flows_1_0[(j-1):j, ...]
                warped_key = backwarp(now_frame_key_f, now_flow, border_mode="REPLICATE")
                now_frame_key_f = F.concat([warped_key, key[j:j+1, ...]], axis=1)
                warped_value = backwarp(now_frame_value_f, now_flow, border_mode="REPLICATE")
                now_frame_value_f = F.concat([warped_value, value[j:j+1, ...]], axis=1)
            # 后面的warp到当前帧
            now_frame_key_b = key[right:right+1, ...]
            now_frame_value_b = value[right:right+1, ...]
            for j in range(right-1, i-1, -1):
                now_flow = flows_0_1[j:j+1, ...]
                warped_key = backwarp(now_frame_key_b, now_flow, border_mode="REPLICATE")
                now_frame_key_b = F.concat([warped_key, key[j:j+1, ...]], axis=1)
                warped_value = backwarp(now_frame_value_b, now_flow, border_mode="REPLICATE")
                now_frame_value_b = F.concat([warped_value, value[j:j+1, ...]], axis=1)

            # 进行concat和pad
            now_frame_key_f = F.concat([now_frame_key_f, now_frame_key_b], axis=1) # [1, (t)*c, h, w]
            now_frame_value_f = F.concat([now_frame_value_f, now_frame_value_b], axis=1) # [1, (t)*c, h, w]
            del now_frame_key_b
            del now_frame_value_b
            _,nums,_,_ = now_frame_key_f.shape
            nums = nums // c
            exp_num = (self.keept+1) * 2
            if nums != exp_num:
                assert 2*nums >= exp_num and nums < exp_num
                now_frame_key_f = F.concat([now_frame_key_f, now_frame_key_f[:, (2*nums - exp_num)*c:nums*c, ...]], axis=1)
                now_frame_value_f = F.concat([now_frame_value_f, now_frame_value_f[:, (2*nums - exp_num)*c:nums*c, ...]], axis=1)
                nums = (self.keept+1) * 2                 
            now_frame_key_f = (now_frame_key_f[:, 0:(nums-1)*c, ...]).reshape(b, nums-1, c, h, w)
            now_frame_value_f = (now_frame_value_f[:, 0:(nums-1)*c, ...]).reshape(b, nums-1, c, h, w)
            if offsets is not None:
                outputs.append(self.time_attention(_query[i:i+1, ...], now_frame_key_f, now_frame_value_f, offsets[i:i+1, ...]))
            else:
                outputs.append(self.time_attention(_query[i:i+1, ...], now_frame_key_f, now_frame_value_f))
        return F.concat(outputs, axis=0) # [b, c, h, w]

    def train_time_attention(self, _query, key, value, t, b, flows_0_1, flows_1_0, offsets, mask_0_1 = None, mask_1_0 = None):
        bt, qkc, h, w = _query.shape
        _, vc, h, w = value.shape
        assert flows_0_1.shape[0] == (t-1) * b
        key = key.reshape(b, t, qkc, h, w)
        value = value.reshape(b, t, vc, h, w)
        
        # 求出每一帧需要attention的内容
        forward_keys = [key[:, 0, ...]] # [B, c, h, w], [B, 2c, h, w], [B, 3c, h, w], [B, 4c, h, w]
        forward_values = [value[:, 0, ...]] # 前两帧分别只有1个和2个，后面每个都是3个
        for i in range(1, t):
            now_frame_key = key[:, i, ...] # [B, c, h, w]
            now_frame_value = value[:, i, ...] # [B, c, h, w]
            now_flow = flows_1_0[(i-1)*b:i*b, ...] # [B, 2, h, w]
            if mask_1_0 is not None:
                self.warp_given_list(forward_keys, now_flow, now_frame_key, dim = qkc, mask=mask_1_0[(i-1)*b:i*b, ...])
                self.warp_given_list(forward_values, now_flow, now_frame_value, dim=vc, mask=mask_1_0[(i-1)*b:i*b, ...])
            else:
                self.warp_given_list(forward_keys, now_flow, now_frame_key, dim = qkc)
                self.warp_given_list(forward_values, now_flow, now_frame_value, dim=vc)
        backward_keys = [key[:, t-1, ...]]
        backward_values = [value[:, t-1, ...]]
        for i in range(t-2, -1, -1):
            now_frame_key = key[:, i, ...] # [B, c, h, w]
            now_frame_value = value[:, i, ...] # [B, c, h, w]
            now_flow = flows_0_1[i*b:(i+1)*b, ...] # [B, 2, h, w]
            if mask_0_1 is not None:
                self.warp_given_list(backward_keys, now_flow, now_frame_key, dim = qkc, mask=mask_0_1[i*b:(i+1)*b, ...])
                self.warp_given_list(backward_values, now_flow, now_frame_value, dim=vc, mask=mask_0_1[i*b:(i+1)*b, ...])
            else:
                self.warp_given_list(backward_keys, now_flow, now_frame_key, dim = qkc)
                self.warp_given_list(backward_values, now_flow, now_frame_value, dim=vc)

        for i in range(t):
            forward_keys[i] = F.concat([forward_keys[i], backward_keys[t-i-1]], axis=1) # [B, (t)*qkc, h, w]
            forward_values[i] = F.concat([forward_values[i], backward_values[t-i-1]], axis=1) # [B, (t)*vc, h, w]
            _,nums,_,_ = forward_keys[i].shape
            nums = nums // qkc
            exp_num = (self.keept+1) * 2
            if nums != exp_num:
                # 补成8个，用后 8 - nums个
                assert 2*nums >= exp_num and nums < exp_num
                forward_keys[i] = F.concat([forward_keys[i], forward_keys[i][:, (2*nums - exp_num)*qkc:nums*qkc, ...]], axis=1)
                forward_values[i] = F.concat([forward_values[i], forward_values[i][:, (2*nums - exp_num)*vc:nums*vc, ...]], axis=1)
                nums = (self.keept+1) * 2                 
            forward_keys[i] = (forward_keys[i][:, 0:(nums-1)*qkc, ...]).reshape(b, nums-1, qkc, h, w)
            forward_values[i] = (forward_values[i][:, 0:(nums-1)*vc, ...]).reshape(b, nums-1, vc, h, w) # delete now_frame  for/back duplicate
        _key = F.stack(forward_keys, axis=1).reshape(bt, self.keept*2+1, qkc, h, w)
        _value = F.stack(forward_values, axis=1).reshape(bt, self.keept*2+1, vc, h, w)
        del forward_keys
        del forward_values
        del backward_keys
        del backward_values
        return self.time_attention(_query, _key, _value, offsets)

    def test_tsm(self, x, t, b, flows_0_1, flows_1_0, mask_0_1 = None, mask_1_0 = None):
        assert b == 1, "for test, make sure batchsize = 1"
        bt, c, h, w = x.shape
        assert bt == t
        key = x
        outputs = [] # 每一帧的结果
        for i in range(t):
            # 确定left和right
            if i < self.keept:
                left = 0
                right = i + self.keept
            elif i >= t - self.keept:
                left = i - self.keept
                right = t - 1
            else:
                left = i - self.keept
                right = i + self.keept

            # 前面的warp到当前帧
            now_frame_key_f = key[left:left+1, ...]
            for j in range(left+1, i+1):
                now_flow = flows_1_0[(j-1):j, ...]
                warped_key = backwarp(now_frame_key_f, now_flow, border_mode="REPLICATE")
                now_frame_key_f = F.concat([warped_key, key[j:j+1, ...]], axis=1)
            # 后面的warp到当前帧
            now_frame_key_b = key[right:right+1, ...]
            for j in range(right-1, i-1, -1):
                now_flow = flows_0_1[j:j+1, ...]
                warped_key = backwarp(now_frame_key_b, now_flow, border_mode="REPLICATE")
                now_frame_key_b = F.concat([warped_key, key[j:j+1, ...]], axis=1)

            # 进行concat和pad
            now_frame_key_f = F.concat([now_frame_key_f, now_frame_key_b], axis=1) # [1, (t)*c, h, w]
            del now_frame_key_b
            _,nums,_,_ = now_frame_key_f.shape
            nums = nums // c
            exp_num = (self.keept+1) * 2
            if nums != exp_num:
                assert 2*nums >= exp_num and nums < exp_num
                now_frame_key_f = F.concat([now_frame_key_f, now_frame_key_f[:, (2*nums - exp_num)*c:nums*c, ...]], axis=1)
                nums = (self.keept+1) * 2                 
            now_frame_key_f = (now_frame_key_f[:, 0:(nums-1)*c, ...])
            outputs.append(self.reduction(now_frame_key_f))
        return F.concat(outputs, axis=0) # [b, c, h, w]

    def train_tsm(self, x, t, b, flows_0_1, flows_1_0, mask_0_1 = None, mask_1_0 = None):
        bt, c, h, w = x.shape
        key = x.reshape(b, t, c, h, w)
        assert flows_0_1.shape[0] == (t-1) * b

        forward_keys = [key[:, 0, ...]] # [B, c, h, w], [B, 2c, h, w], [B, 3c, h, w], [B, 4c, h, w]
        for i in range(1, t):
            now_frame_key = key[:, i, ...] # [B, c, h, w]
            now_flow = flows_1_0[(i-1)*b:i*b, ...] # [B, 2, h, w]
            if mask_1_0 is not None:
                self.warp_given_list(forward_keys, now_flow, now_frame_key, dim = c, mask=mask_1_0[(i-1)*b:i*b, ...])
            else:
                self.warp_given_list(forward_keys, now_flow, now_frame_key, dim = c)
        
        backward_keys = [key[:, t-1, ...]]
        for i in range(t-2, -1, -1):
            now_frame_key = key[:, i, ...] # [B, c, h, w]
            now_flow = flows_0_1[i*b:(i+1)*b, ...] # [B, 2, h, w]
            if mask_0_1 is not None:
                self.warp_given_list(backward_keys, now_flow, now_frame_key, dim = c, mask=mask_0_1[i*b:(i+1)*b, ...])
            else:
                self.warp_given_list(backward_keys, now_flow, now_frame_key, dim = c)

        for i in range(t):
            forward_keys[i] = F.concat([forward_keys[i], backward_keys[t-i-1]], axis=1) # [B, (t)*c, h, w]
            _,nums,_,_ = forward_keys[i].shape
            nums = nums // c
            exp_num = (self.keept+1) * 2
            if nums != exp_num:
                assert 2*nums >= exp_num and nums < exp_num
                forward_keys[i] = F.concat([forward_keys[i], forward_keys[i][:, (2*nums - exp_num)*c:nums*c, ...]], axis=1)
                nums = (self.keept+1) * 2
            forward_keys[i] = (forward_keys[i][:, 0:(nums-1)*c, ...]) # delete now_frame  for/back duplicate
        _key = F.stack(forward_keys, axis=1).reshape(bt, -1, h, w)
        del forward_keys
        del backward_keys
        return self.reduction(_key)

    def forward(self, x, t, b, flows_0_1, flows_1_0, offsets, mask_0_1, mask_1_0):
        """
            flows_0_1: [(T-1)*B, 2, h, w]
        """
        if self.headnums > 0:
            _query = self.do_q_k_v_outputlinear(x, t, self.query_embedding) # [bt, c, h, w]
            _key = self.do_q_k_v_outputlinear(x, t, self.key_embedding)
            _value = self.do_q_k_v_outputlinear(x, t, self.value_embedding)
            if t > test_t_thr:
                outputs = self.test_time_attention(_query, _key, _value, t, b, flows_0_1, flows_1_0, offsets, mask_0_1, mask_1_0)
            else:
                outputs = self.train_time_attention(_query, _key, _value, t, b, flows_0_1, flows_1_0, offsets, mask_0_1, mask_1_0)
        else:
            if t > test_t_thr:
                outputs = self.test_tsm(x, t, b, flows_0_1, flows_1_0, mask_0_1, mask_1_0)
            else:
                outputs = self.train_tsm(x, t, b, flows_0_1, flows_1_0, mask_0_1, mask_1_0)
        outputs = self.do_q_k_v_outputlinear(outputs, t, self.output_linear)
        return outputs

    def init_weights(self):
        if self.layer_norm:
            default_init_weights(self.output_linear)
        else:
            default_init_weights(self.output_linear, nonlinearity='leaky_relu', lrelu_value=0.1)

        if self.headnums == 0:
            default_init_weights(self.reduction)

class TransformerBlock(M.Module):
    """
        Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """
    def __init__(self, heads, hidden, layer_norm, layer_id, keept, offsets_num, do_not_attention):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads, hidden, layer_norm, layer_id, keept, offsets_num)
        self.feed_forward = FeedForward(hidden, layer_norm)
        self.do_not_attention = do_not_attention

    def forward(self, Dict):
        x = Dict['x']
        t = Dict['t']
        b = Dict['b']
        flows_0_1 = Dict['flows_0_1']
        flows_1_0 = Dict['flows_1_0']
        mask_1_0 = Dict['mask_1_0']
        mask_0_1 = Dict['mask_0_1']
        offsets = Dict['offsets']
        if not self.do_not_attention:
            x = x + self.attention(x, t, b, flows_0_1, flows_1_0, offsets, mask_0_1, mask_1_0)
        if t > test_t_thr: # test
            res = []
            for i in range(t):
                res.append(self.feed_forward(x[i:(i+1), ...]))
            x = F.concat(res, axis=0)
        else:
            x = self.feed_forward(x)
        return {'x':x, 'b':b, 't':t, "flows_0_1": flows_0_1, "flows_1_0": flows_1_0, "offsets": offsets, "mask_1_0":mask_1_0, "mask_0_1":mask_0_1}

@BACKBONES.register_module()
class FTVSR(M.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 channels = 16,
                 layers = 6,
                 reconstruction_blocks = 4,
                 flownet_layers = 6,
                 pretrained_optical_flow_path = None,
                 heads = 4,
                 upscale_factor=4,
                 layer_norm = False,
                 keept = 2,
                 learned_offsets_num = 2, # <=0 表示不使用，只做相邻帧的attention
                 use_flow_mask = False,
                 do_not_attention = False): 
        super(FTVSR, self).__init__()
        self.upscale_factor = upscale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_norm = layer_norm
        if heads > 0:
            self.channels = channels * heads
        else:
            self.channels = channels
        self.learned_offsets_num = learned_offsets_num
        self.use_flow_mask = use_flow_mask
        self.flow_thr = 2

        self.flownet = Spynet(num_layers=flownet_layers, pretrain_ckpt_path=pretrained_optical_flow_path, blocktype = "resblock")
        if learned_offsets_num > 0:
            self.offsetnet = Spynet2(channels=self.channels, offsets_num=learned_offsets_num)

        blocks = []
        for idx in range(layers):
            blocks.append(TransformerBlock(heads = heads, hidden = self.channels, 
                                           layer_norm = layer_norm, layer_id = idx, keept = keept, 
                                           offsets_num = learned_offsets_num, do_not_attention = do_not_attention))

        self.transformer = M.Sequential(*blocks)

        self.conv1 = M.Conv2d(self.in_channels, self.channels, kernel_size = 3, stride=1, padding=1)
        self.encoder = M.Sequential(
            FeedForward(self.channels, layer_norm=False),
            FeedForward(self.channels, layer_norm=False)
        )

        self.reconstruction = ResBlocks(channel_num=self.channels, resblock_num=reconstruction_blocks)    
        self.upsample1 = PixelShufflePack(self.channels, self.channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(self.channels, 64, 2, upsample_kernel=3)
        self.conv_hr = M.Conv2d(64, 64, 3, 1, 1)  # need init
        self.conv_last = M.Conv2d(64, out_channels, 3, 1, 1)

        self.lrelu = M.LeakyReLU(negative_slope=0.1)

    def do_upsample(self, x):
        out = self.reconstruction(x)
        out = self.lrelu(self.upsample1(out))
        out = self.lrelu(self.upsample2(out))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        return out

    def get_flow_mask(self, flows_0_1, flows_1_0):
        if self.use_flow_mask:
            mask_1_0 = cal_flow_mask(flows_1_0, flows_0_1, thredhold=self.flow_thr)
            mask_0_1 = cal_flow_mask(flows_0_1, flows_1_0, thredhold=self.flow_thr)
            return mask_1_0, mask_0_1
        else:
            return None, None

    def test_forward(self, frames, flows_0_1, flows_1_0):
        b, t, c, h, w = frames.shape # [1, 100 ,c, h, w]
        assert b == 1
        enc_feat = []
        offsets = []
        for i in range(t):
            enc_feat.append(self.encoder(self.conv1(frames[0, i:(i+1), ...])))
            if self.learned_offsets_num > 0:
                offsets.append(self.offsetnet(enc_feat[-1])) # 取最新的
        enc_feat = F.concat(enc_feat, axis=0)
        if len(offsets) > 0:
            offsets = F.concat(offsets, axis=0)
        else:
            offsets = None
        mask_1_0, mask_0_1 = self.get_flow_mask(flows_0_1, flows_1_0)
        enc_feat = self.transformer({"x":enc_feat, "t":t, "b":b, "flows_0_1": flows_0_1, "flows_1_0": flows_1_0, 
                                     "offsets": offsets, "mask_1_0":mask_1_0, "mask_0_1":mask_0_1})['x'] # [bt,c,h,w]
        res = []
        for i in range(t):
            res.append(self.do_upsample(enc_feat[i:(i+1), ...]))
        res = F.concat(res, axis=0) # [100,3,4h, 4w]
        return F.expand_dims(res, axis=0) # [1,100,3,4h,4w]

    def forward(self, frames, flows_0_1, flows_1_0):
        b, t, c, h, w = frames.shape
        offsets = None
        enc_feat = self.encoder(self.conv1(frames.reshape(b*t, c, h, w)))
        if self.learned_offsets_num > 0:
            offsets = self.offsetnet(enc_feat) # [bt, heads, num, 2, h, w]
        mask_1_0, mask_0_1 = self.get_flow_mask(flows_0_1, flows_1_0)
        enc_feat = self.transformer({"x":enc_feat, "t":t, "b":b, "flows_0_1": flows_0_1, "flows_1_0": flows_1_0, 
                                     "offsets": offsets, "mask_1_0":mask_1_0, "mask_0_1":mask_0_1})['x'] # [bt,c,h,w]
        enc_feat = self.do_upsample(enc_feat)
        return enc_feat.reshape(b, t, self.out_channels, h*self.upscale_factor, w*self.upscale_factor)

    def init_weights(self, pretrained=None, strict=True):
        self.flownet.init_weights(strict=False)
        default_init_weights(self.conv1)
        default_init_weights(self.upsample1, nonlinearity='leaky_relu', lrelu_value=0.1)
        default_init_weights(self.upsample2, nonlinearity='leaky_relu', lrelu_value=0.1)
        default_init_weights(self.conv_hr, nonlinearity='leaky_relu', lrelu_value=0.1)

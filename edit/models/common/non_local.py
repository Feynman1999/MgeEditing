import numpy as np
import megengine
import megengine.module as M
import megengine.functional as F
import math

def do_attention(query, key, value = None):
    scores = F.matmul(query, key) / math.sqrt(query.shape[-1]) # b, t*out_h*out_w, t*out_h*out_w
    p_attn = F.nn.softmax(scores, axis=2)
    if value is None:
        return p_attn
    p_val = F.matmul(p_attn, value)
    return p_val, p_attn

class PAM_Module(M.Module):
    """ Position attention module"""
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = M.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv = M.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv = M.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = megengine.Parameter(np.zeros((1,), dtype=np.float32))

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.shape
        proj_query = self.query_conv(x).reshape(m_batchsize, -1, width * height).transpose(0, 2, 1)
        proj_key = self.key_conv(x).reshape(m_batchsize, -1, width * height)
        proj_value = self.value_conv(x).reshape(m_batchsize, -1, width * height).transpose(0, 2, 1)
        out, _ = do_attention(proj_query, proj_key, proj_value)
        out = out.transpose(0, 2, 1).reshape(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out

class CAM_Calculate(M.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Calculate, self).__init__()
        self.chanel_in = in_dim

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.shape
        proj_query = x.reshape(m_batchsize, C, -1)
        proj_key = x.reshape(m_batchsize, C, -1).transpose(0, 2, 1)
        
        attention = do_attention(proj_query, proj_key, value=None)

        return attention

class CAM_Use(M.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Use, self).__init__()
        self.chanel_in = in_dim
        self.gamma = megengine.Parameter(np.zeros((1,), dtype=np.float32))

    def forward(self, x, attention):
        """
            inputs :
                x : input feature maps( B X C X H X W)
                attention: B X C X C
            returns :
                out : attention value + input feature
        """
        m_batchsize, C, height, width = x.shape
        proj_value = x.reshape(m_batchsize, C, -1)

        out = F.matmul(attention, proj_value)
        out = out.reshape(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out

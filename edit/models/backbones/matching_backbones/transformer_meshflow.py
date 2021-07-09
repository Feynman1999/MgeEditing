import numpy as np
import megengine
import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
from edit.models.builder import BACKBONES
from edit.models.common import add_H_W_Padding, default_init_weights, ResBlock, ResBlocks, PixelShufflePack, DLT_solve
from edit.models.losses import CharbonnierLoss
import math

class FeedForward(M.Module):
    def __init__(self, d_model, layer_norm = False):
        super(FeedForward, self).__init__()
        self.layer_norm = layer_norm
        if layer_norm:
            self.conv = M.Sequential(
                M.Conv2d(d_model, d_model, kernel_size=3, padding=1, dilation=1, stride=1),
                M.normalization.LayerNorm(d_model),
                M.ReLU(),
                M.Conv2d(d_model, d_model, kernel_size=3, padding=1, dilation=1, stride=1),
            )
            self.init_weights()
        else:
            self.conv = ResBlock(d_model, d_model, init_scale=0.1)

    def forward(self, x):
        if self.layer_norm:
            return x + self.conv(x)
        else:
            return self.conv(x)
    
    def init_weights(self):
        default_init_weights(self.conv, scale=0.1)

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
        self.init_weights()

    def forward(self, x, t, b):
        bt, c, h, w = x.shape
        d_k = c // self.headnums  # 每个head的通道数
        outputs = []
        _query = self.query_embedding(x)
        _key = self.key_embedding(x)
        _value = self.value_embedding(x)
        for idx in range(self.headnums):
            height, width = 9, 16
            query, key, value = _query[:, (idx*d_k):(idx*d_k + d_k), ...], _key[:, (idx*d_k):(idx*d_k + d_k), ...], _value[:, (idx*d_k):(idx*d_k + d_k), ...]
            out_h, out_w = h // height, w // width
            # 1) embedding and reshape
            query = query.reshape(b, t, d_k, out_h, height, out_w, width)
            query = query.transpose((0, 1, 3, 5, 2, 4, 6)).reshape((b,  t*out_h*out_w, d_k*height*width))
            key = key.reshape(b, t, d_k, out_h, height, out_w, width)
            key = key.transpose((0, 1, 3, 5, 2, 4, 6)).reshape((b,  t*out_h*out_w, d_k*height*width))
            value = value.reshape(b, t, d_k, out_h, height, out_w, width)
            value = value.transpose((0, 1, 3, 5, 2, 4, 6)).reshape((b,  t*out_h*out_w, d_k*height*width))
            # 2) Apply attention on all the projected vectors in batch.
            y, _ = do_attention(query, key, value)
            # 3) "Concat" using a view and apply a final linear.
            # print(y.shape)
            # print(b, t, out_h, out_w, d_k, height, width)
            y = y.reshape(b, t, out_h, out_w, d_k, height, width)
            y = y.transpose((0, 1, 4, 2, 5, 3, 6)).reshape(bt, d_k, h, w)
            outputs.append(y)
        outputs = F.concat(outputs, axis = 1)
        return self.output_linear(outputs)

    def init_weights(self):
        default_init_weights(self.output_linear, scale=0.1)

class TransformerBlock(M.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """
    def __init__(self, heads, hidden, layer_norm):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads, hidden, layer_norm)
        self.feed_forward = FeedForward(hidden, layer_norm)

    def forward(self, Dict):
        x = Dict['x']
        t = Dict['t']
        b = Dict['b']
        x = x + self.attention(x, t, b)
        x = self.feed_forward(x)
        return {'x':x, 'b':b, 't':t}


@BACKBONES.register_module()
class TMF(M.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 channels = 16,
                 layers = 6,
                 heads = 4,
                 layer_norm = True):
        super(TMF, self).__init__()
        self.blocks = [9, 16]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_norm = layer_norm
        self.channels = channels * heads
        self.charloss = CharbonnierLoss()

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

        self.downsample1 = M.Sequential(
            M.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=(3,4), stride=(3, 4), padding=1),
            M.LayerNorm(self.channels),
            M.ReLU()
        ) # stride 3, 4 [B,C,h,W]
        self.conv_last1 = ResBlocks(channel_num=self.channels, resblock_num=2)
        self.downsample2 = M.Sequential(
            M.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=2, stride=(3, 4), padding=1),
            M.LayerNorm(self.channels),
            M.ReLU()
        ) # stride 3, 4 [B,C,h,w]
        self.conv_last2 = M.Conv2d(self.channels, self.out_channels, 3, 1, 1)
        
    def do_downsample(self, x):
        out = self.downsample1(x)
        out = self.conv_last1(out)
        out = self.downsample2(out)
        out = self.conv_last2(out)
        return out

    def forward(self, frames):
        b, t, c, h, w = frames.shape # t = 2
        enc_feat = self.encoder(self.conv1(frames.reshape(b*t, c, h, w)))
        enc_feat = self.transformer({"x":enc_feat, "t":t, "b":b})['x'] # [bt,c,h,w]
        enc_feat = self.do_downsample(enc_feat)
        return enc_feat.reshape(b, t, self.out_channels, h//self.blocks[0] + 1, w//self.blocks[1] + 1)

    def init_weights(self, pretrained=None, strict=True):
        default_init_weights(self.conv1)
        default_init_weights(self.downsample1)
        default_init_weights(self.downsample2)
        default_init_weights(self.conv_last2)

    def get_denseflow_by_meshflow(self, meshflow, block_h, block_w):
        """
            解线性方程（求逆矩阵）
            block_h, block_w    e.g.   9, 16
            meshflow: (B, 2, h, w) e.g.  h=11, w=11
            return: (B, H, W, 2) e.g.  H = (h-1)*block_h,  W = (w-1)* block_w
        """
        # 对meshflow做repeat
        B,C,h,w = meshflow.shape
        H = (h-1) * block_h # train: 90
        W = (w-1) * block_w # train: 160
        assert C == 2
        meshflow = F.repeat(meshflow, 2, axis = 2)
        meshflow = F.repeat(meshflow, 2, axis = 3) # [B, 2, 22, 22]
        meshflow = meshflow[:, :, 1:1+2*(h-1), 1:1+2*(w-1)] # [B, 2, 20, 20]
        # reshape meshflow to 2*2
        meshflow = meshflow.reshape(B, 2, h-1, 2, w-1, 2)
        # get many points flow
        meshflow = meshflow.transpose(0,2,4,3,5,1).reshape(B*(h-1)*(w-1), 4, 2) # 分别表示左右（0， x）和上下（1，y）方向的运动

        # get 对应的坐标  shape (x, 4, 2)
        x_list = np.arange(0., W - 1., block_w).reshape(1, w-1, 1) # train: 10个
        y_list = np.arange(0., H - 1., block_h).reshape(h-1, 1, 1) # train: 10个
        x_list = x_list.repeat(h-1, axis=0) # (10,10, 1)
        y_list = y_list.repeat(w-1, axis=1) # (10,10, 1)
        xy_list_up_left = np.concatenate((x_list, y_list), 2).reshape(-1, 2)  # [100,2]
        xy_list_up_right = np.concatenate((x_list+block_w-1, y_list), 2).reshape(-1, 2)  # [100,2]
        xy_list_down_left = np.concatenate((x_list, y_list + block_h-1), 2).reshape(-1, 2)  # [100,2]
        xy_list_down_right = np.concatenate((x_list + block_w - 1, y_list + block_h-1), 2).reshape(-1, 2)  # [100,2]
        # 按照顺序stack成[100,4,2]
        xy_list = np.stack([xy_list_up_left, xy_list_up_right, xy_list_down_left, xy_list_down_right], axis=1) # [100,4,2]
        # repeat batch times
        xy_list = np.tile(xy_list, (B, 1, 1)) # (2400, 4, 2)
        # solve matrix
        matrix = DLT_solve(src_ps = megengine.tensor(xy_list.astype(np.float32)), off_sets = meshflow) # x,3,3

        # 对大的坐标(x, block_h*block_w, 2) 做3*3矩阵乘法，得到应该的sample坐标
        dense_x_list = np.linspace(0., W - 1., W).reshape(1, W, 1)
        dense_y_list = np.linspace(0., H - 1., H).reshape(H, 1, 1)
        dense_x_list = dense_x_list.repeat(H, axis=0)
        dense_y_list = dense_y_list.repeat(W, axis=1)
        dense_xy_list = np.concatenate((dense_x_list, dense_y_list), 2) # [H, W, 2]
        dense_xy_list = dense_xy_list.reshape((h-1), block_h, (w-1), block_w, 2)
        dense_xy_list = dense_xy_list.transpose(0, 2, 1, 3, 4).reshape((h-1)*(w-1), block_h*block_w, 2)
        dense_xy_list = np.tile(dense_xy_list, (B, 1, 1))  # (x, block_h*block_w, 2)
        BX, points, _ = dense_xy_list.shape
        dense_xy_list = np.concatenate([dense_xy_list, np.ones((BX, points, 1))*1.0], axis=2).transpose(0,2,1) 
        dense_xy_list = megengine.tensor(dense_xy_list.astype(np.float32)) # (x, 3, block_h*block_w)
        # solve new grid
        grid = F.matmul(matrix, dense_xy_list) # [x, 3, block_h*block_w]
        grid = grid.transpose(0,2,1)[:, :, 0:2] # [x, block_h*block_w, 2]
        grid = grid.reshape(B, (h-1), (w-1), block_h, block_w, 2)
        grid = grid.transpose(0, 1, 3, 2, 4, 5).reshape(B, H, W, 2)
        return grid

    def loss(self, meshflow, image):
        """
            meshflow: B,2,2,H,W
            image: B,2,3,H,W
        """
        B,T,C,h,w = meshflow.shape
        assert C == 2, "channel of flow should be 2"
        assert T == 2, "now should input 2 images"
        grid = self.get_denseflow_by_meshflow(meshflow.reshape(-1, C, h, w), self.blocks[0], self.blocks[1])
        grid = grid.reshape(B, T, (h-1)*self.blocks[0], (w-1)*self.blocks[1], C)
        
        # 0 -> 1 flow
        grid_0_1 = grid[:, 0, ...] # [B,H,W,2]
        warp_1_0 = F.vision.remap(inp = image[:, 1, ...], map_xy=grid_0_1, border_mode="CONSTANT")
        loss_0_1 = self.charloss(warp_1_0, image[:, 0, ...])

        # 1 -> 0 flow
        grid_1_0 = grid[:, 1, ...] # [B,H,W,2]
        warp_0_1 = F.vision.remap(inp = image[:, 0, ...], map_xy=grid_1_0, border_mode="CONSTANT")
        loss_1_0 = self.charloss(warp_0_1, image[:, 1, ...])

        # total loss
        loss = (loss_0_1 + loss_1_0) * 0.5
        return loss

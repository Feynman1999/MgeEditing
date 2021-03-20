import numpy as np
import megengine as mge
import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
from edit.models.common import ResBlocks
from edit.models.builder import BACKBONES


@BACKBONES.register_module()
class DeepMeshFlow(M.Module):
    def __init__(self, in_channels,
                       channels = 32,
                       feature_blocknums = 5,
                       mask_blocknums = 3,
                       aggr_blocknums = 10,
                       blocktype = "resblock"):
        super(DeepMeshFlow, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.blocknums = blocknums
        self.blocktype = blocktype

        self.conv1 = M.ConvRelu2d(in_channels, channels, kernel_size=3, stride=1, padding=1) # need init
        self.conv_feature = ResBlocks(channel_num=channels, resblock_num=feature_blocknums, blocktype=blocktype) # share
        self.conv_aggr = ResBlocks(channel_num=2*channels, resblock_num=aggr_blocknums, blocktype=blocktype)

        self.conv_mask = M.Sequential(
            M.ConvBnRelu2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False),
            M.ConvBnRelu2d(4, 8, kernel_size=3, stride=1, padding=1, bias=False),
            M.ConvBnRelu2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
            M.ConvBnRelu2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            M.ConvBn2d(32, 1, kernel_size=3, padding=1, bias=False),
            M.Sigmoid()
        )

    def get_feature(self, img):
        return self.conv_feature(self.conv1(img))

    def get_mask(self, img):
        return self.conv_mask(img)  # [B,1,h,w]

    def forward(self, img1, img2):
        """
            return:  pred_img2  (warped by img1)
                     mask of img2
                     mask of pred_img2
        """
        img1_fea = self.get_feature(img1)
        img2_fea = self.get_feature(img2)
        img1_mask = self.get_mask(img1)
        img2_mask = self.get_mask(img2)
        img1_fea = img1_fea * img1_mask
        img2_fea = img2_fea * img2_mask
        fea = self.conv_aggr(F.concat([img1_fea, img2_fea], axis=1)) # [B,2c,h,w]
        # 三个分支，目前就搞中间一个，看看收敛效果
        # 生成一个（18+1）*（32+1）的mesh，然后根据这个mesh获得 flow
        # 对于19 * 33的mesh，解19*33个变换，然后作用到
        

    def get_loss(self, include_mask = False):
        pass
    
    def init_weights(self, pretrained=None, strict=True):
        pass

import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
from edit.models.builder import BACKBONES


# 如果一个模块需要用自己的初始化，就写一个方法，在全局初始化之后，会调用它
class UPU(M.Module):
    """up-projection units"""

    def __init__(self,
                 num_channels,
                 filter_size,
                 stride,
                 padding):
        super(UPU, self).__init__()
        self.deconv1 = M.Sequential(
            ConvTranspose2d(num_channels, num_channels, filter_size, stride, padding),
            M.PReLU(num_parameters=1, init=0.25)
        )
        self.conv1 = M.Sequential(
            Conv2d(num_channels, num_channels, filter_size, stride, padding),
            M.PReLU(num_parameters=1, init=0.25)
        )
        self.deconv2 = M.Sequential(
            ConvTranspose2d(num_channels, num_channels, filter_size, stride, padding),
            M.PReLU(num_parameters=1, init=0.25)
        )

    def forward(self, x1):
        x2 = self.deconv1(x1)
        x3 = self.conv1(x2)
        x3 = x3 - x1
        x4 = self.deconv2(x3)
        return x4 + x2


class DPU(M.Module):
    """
        down-projection units
    """
    def __init__(self, num_channels, filter_size, stride, padding):
        super(DPU, self).__init__()
        self.conv1 = M.Sequential(
            Conv2d(num_channels, num_channels, filter_size, stride, padding),
            M.PReLU(num_parameters=1, init=0.25)
        )
        self.deconv1 = M.Sequential(
            ConvTranspose2d(num_channels, num_channels, filter_size, stride, padding),
            M.PReLU(num_parameters=1, init=0.25)
        )
        self.conv2 = M.Sequential(
            Conv2d(num_channels, num_channels, filter_size, stride, padding),
            M.PReLU(num_parameters=1, init=0.25)
        )

    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = self.deconv1(x2)
        x3 = x3 - x1
        x4 = self.conv2(x3)
        return x4 + x2


@BACKBONES.register_module()
class DBPN(M.Module):
    """DBPN network structure.

    Paper:
    Ref repo:

    Args:
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 n_0=256,
                 n_R=64,
                 iterations_num=10,
                 upscale_factor=4):
        super(DBPN, self).__init__()

        filter_size = upscale_factor + 4
        stride = upscale_factor
        padding = 2
        self.iterations_num = iterations_num
        self.conv1 = M.Sequential(
            Conv2d(in_channels, n_0, 3, 1, 1),
            M.PReLU(num_parameters=1, init=0.25)
        )
        self.conv2 = M.Sequential(
            Conv2d(n_0, n_R, 1, 1, 0),
            M.PReLU(num_parameters=1, init=0.25)
        )
        self.UPU = UPU(n_R, filter_size, stride, padding)
        self.DPU = DPU(n_R, filter_size, stride, padding)
        self.conv3 = M.Sequential(
            Conv2d(n_R*iterations_num, out_channels, 3, 1, 1),
            M.PReLU(num_parameters=1, init=0.25)
        )

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        H_list = []
        x = self.conv1(x)
        x = self.conv2(x)

        for _ in range(self.iterations_num-1):
            H = self.UPU(x)
            H_list.append(H)
            x = self.DPU(H)

        H_list.append(self.UPU(x))
        x = F.concat(H_list, axis=1)
        x = self.conv3(x)
        return x

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

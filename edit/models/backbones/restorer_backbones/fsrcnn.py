import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
from edit.models.builder import BACKBONES


@BACKBONES.register_module()
class FSRCNN(M.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 d = 56,
                 s = 12,
                 upscale_factor=4):
        super(FSRCNN, self).__init__()

        l = []
        l.append(M.Sequential(
            Conv2d(in_channels, d, 5, 1, 2),
            M.PReLU(num_parameters=1, init=0.25)
        ))
        l.append(M.Sequential(
            Conv2d(d, s, 1, 1, 0),
            M.PReLU(num_parameters=1, init=0.25)
        ))
        for i in range(4):
            l.append(M.Sequential(
                Conv2d(s, s, 3, 1, 1),
                M.PReLU(num_parameters=1, init=0.25)
            ))
        l.append(M.Sequential(
            Conv2d(s, d, 1, 1, 0),
            M.PReLU(num_parameters=1, init=0.25)
        ))
        l.append(ConvTranspose2d(d, out_channels, 8, upscale_factor, padding=2))
        self.convs = M.Sequential(*l)

    def forward(self, x):
        x = self.convs(x)
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

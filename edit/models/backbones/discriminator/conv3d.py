import megengine.module as M
import megengine.functional as F
from edit.models.builder import BACKBONES
from .spectral_norm import use_spectral_norm


@BACKBONES.register_module()
class Discriminator(M.Module):
    def __init__(self, in_channels=4, use_sigmoid=False, use_sn=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 36
        # 时序3 空间5
        self.conv = M.Sequential(
            use_spectral_norm(M.Conv3d(in_channels=in_channels, out_channels=nf*1, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=1, bias=not use_sn), use_sn),

            M.LeakyReLU(0.2),
            use_spectral_norm(M.Conv3d(nf*1, nf*2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_sn), use_sn),

            M.LeakyReLU(0.2),
            use_spectral_norm(M.Conv3d(nf * 2, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_sn), use_sn),

            M.LeakyReLU(0.2),
            use_spectral_norm(M.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_sn), use_sn),

            M.LeakyReLU(0.2),
            use_spectral_norm(M.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_sn), use_sn),

            M.LeakyReLU(0.2),
            M.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5),
                      stride=(1, 2, 2), padding=(1, 2, 2))
        )

    def forward(self, xs):
        # BT, C, H, W = xs.shape
        xs_t = xs.transpose(1, 0, 2, 3)
        xs_t = F.expand_dims(xs_t, axis=0)
        feat = self.conv(xs_t) # 1, C, BT, H, W
        if self.use_sigmoid:
            feat = F.sigmoid(feat)  
        return feat.transpose(0, 2, 1, 3, 4)  # 1, BT, C, H, W

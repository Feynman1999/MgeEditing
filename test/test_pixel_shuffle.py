import numpy as np
import megengine as mge
import megengine.module as M
mge.set_default_device('gpu1')

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

net = PixelShuffle(2)
x = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
x = np.array(x)
res = net(mge.tensor(x).transpose(0, 3, 1, 2))
print(res)
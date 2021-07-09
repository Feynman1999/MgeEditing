from collections import defaultdict
from megengine.module.normalization import GroupNorm
import numpy as np
import megengine
import megengine.module as M
import megengine.functional as F
import math
from . import default_init_weights

class BasicBlock(M.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = M.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn1 = M.GroupNorm(8, planes)
        self.relu = M.ReLU()
        self.conv2 = M.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = M.GroupNorm(8, planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(M.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = M.Conv2d(inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = M.GroupNorm(8, bottle_planes)
        self.conv2 = M.Conv2d(bottle_planes, bottle_planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = M.GroupNorm(8, bottle_planes)
        self.conv3 = M.Conv2d(bottle_planes, planes, kernel_size=1, bias=False)
        self.bn3 = M.GroupNorm(8, planes)
        self.relu = M.ReLU()
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)
        return out


class Root(M.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = M.Conv2d(in_channels, out_channels, 1, stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = M.GroupNorm(8, out_channels)
        self.relu = M.ReLU()
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(F.concat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)
        return x

class Tree(M.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1, level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = M.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = M.Sequential(
                M.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                M.GroupNorm(8, out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x

class DLA(M.Module):
    def __init__(self, levels, channels, num_classes=1, block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes

        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1])
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2, level_root=False, root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2, level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2, level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2, level_root=True, root_residual=residual_root)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                M.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                M.GroupNorm(8, planes),
                M.ReLU()])
            inplanes = planes
        return M.Sequential(*modules)

    def forward(self, x):
        y = []
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        # 此处相对于dla34的返回结果进行了变动，返回6个level的特征图
        return y


def dla34(**kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [32, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    return model


def fill_up_weights(up):
    w = up.weight
    # print(w.shape) #   (groups, in_channels // groups, out_channels // groups, height, width)
    f = math.ceil(w.shape[3] / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.shape[3]):
        for j in range(w.shape[4]):
            w[0, 0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.shape[0]):
        w[c, 0, 0, :, :] = w[0, 0, 0, :, :]


class DeformConv(M.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = M.Sequential(
            M.GroupNorm(8, cho),
            M.ReLU()
        )
        channels_ = 3 * 3 * 3
        self.conv_offset_mask = M.Conv2d(chi,
                                    channels_,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=True)
        self.deformconv = M.DeformableConv2d(chi, cho, kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, x):
        # get offset and mask
        out = self.conv_offset_mask(x)
        # out: b,27,h,w
        offset = out[:, 0:18, :, :]
        mask = out[:, 18:, :, :]
        mask = F.sigmoid(mask)
        x = self.deformconv(x, offset, mask)
        x = self.actf(x)
        return x


class IDAUp(M.Module):
    '''
    IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j])
    ida(layers, len(layers) -i - 2, len(layers))
    '''
    def __init__(self, o, channels, up_f):
        # j = -3
        # o: 128 
        # channels: [128,256,512]
        # up_f: [2/2, 4/2,8/2] = [1,2,4]
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)): # i in 1, 2
            c = channels[i]
            f = int(up_f[i])  # 2 
            proj = DeformConv(c, o)
            node = DeformConv(o, o)

            up = M.ConvTranspose2d(o, o, f *2, stride=f, padding=f//2, groups=1, bias=False)
            # fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
                 
    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):# i in 4, 5
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))

            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(M.Module):
    '''
    # first_level = 2 if down_ratio=4
    # channels = [64, 128, 256, 512]
    # scales = [1, 2, 4, 8]
    '''
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)

        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i), IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class DLASeg_GN(M.Module):
    def __init__(self, num_layers, inp, pretrained):
        super(DLASeg_GN, self).__init__()
        self.first_level = 2
        self.last_level = 5

        self.base = dla34()
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))] # [1,2,4,8]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)
        out_channel = channels[self.first_level]

        # 进行上采样
        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], 
                            [2 ** i for i in range(self.last_level - self.first_level)])
        self.init_weights()

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i])

        self.ida_up(y, 0, len(y))

        return y[-1]

    def init_weights(self):
        default_init_weights(self, scale=0.2)

from megengine.module import LocalConv2d
import megengine.functional as F
from megengine import Parameter, tensor
import numpy as np


def test_local_conv2d():
    batch_size = 10
    in_channels = 4
    out_channels = 8
    input_height = 64
    input_width = 64
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    groups = 1
    local_conv2d = LocalConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    inputs = np.random.normal(
        size=(batch_size, in_channels, input_height, input_width)
    ).astype(np.float32)
    output_height = (input_height + padding * 2 - kernel_size) // stride + 1
    output_width = (input_width + padding * 2 - kernel_size) // stride + 1
    weights = np.random.normal(
        size=(
            groups,
            output_height,
            output_width,
            in_channels // groups,
            kernel_size,
            kernel_size,
            out_channels // groups,
        )
    ).astype(np.float32)
    local_conv2d.weight = Parameter(weights)
    outputs = local_conv2d(tensor(inputs))
    return outputs

output = test_local_conv2d()
print(output.shape)
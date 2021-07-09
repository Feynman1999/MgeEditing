import megengine.module as M
import megengine.functional as F

def default_init_weights(module, scale=1, nonlinearity="relu", lrelu_value = 0.1):
    """
        nonlinearity: leaky_relu
    """
    for m in module.modules():
        if isinstance(m, M.Conv2d):
            M.init.msra_normal_(m.weight, a=lrelu_value, mode="fan_in", nonlinearity=nonlinearity)
            m.weight *= scale
            if m.bias is not None:
                M.init.zeros_(m.bias)
        elif isinstance(m, M.ConvTranspose2d):
            M.init.normal_(m.weight, 0, 0.001)
            m.weight *= scale
            if m.bias is not None:
                M.init.zeros_(m.bias)
        else:
            pass


def gaussian2D(radius, sigma=1):
    """Generate 2D gaussian kernel.
    Args:
        radius (int): Radius of gaussian kernel.
        sigma (int): Sigma of gaussian function. Default: 1.
    Returns:
        h (Tensor): Gaussian kernel with a
            ``(2 * radius + 1) * (2 * radius + 1)`` shape.
    """
    x = F.arange(-radius, radius + 1).reshape(1, -1)
    y = F.arange(-radius, radius + 1).reshape(-1, 1)
    h = F.exp((-(x * x + y * y) / (2 * sigma * sigma)))
    h[h < 1.1920928955e-07 * h.max()] = 0.0
    return h


def gen_gaussian_target(heatmap, center, radius, k=1):
    """Generate 2D gaussian heatmap.
    Args:
        heatmap (Tensor): Input heatmap, the gaussian kernel will cover on
            it and maintain the max value.
        center (list[int]): Coord of gaussian kernel's center.
        radius (int): Radius of gaussian kernel.
        k (int): Coefficient of gaussian kernel. Default: 1.
    Returns:
        out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
    """
    diameter = 2 * radius + 1
    gaussian_kernel = gaussian2D(radius, sigma=diameter / 6)
    height, width = heatmap.shape[:2]

    x, y = center
    x = max(0, x)
    y = max(0, y)
    x = min(width-1, x)
    y = min(height-1, y)
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_gaussian = gaussian_kernel[radius - top:radius + bottom, radius - left:radius + right]
    assert heatmap.dtype == masked_gaussian.dtype
    heatmap[y - top:y + bottom, x - left:x + right] = F.maximum(heatmap[y - top:y + bottom, x - left:x + right], masked_gaussian * k)
    return heatmap
    
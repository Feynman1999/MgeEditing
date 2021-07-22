import megengine
import numpy as np
import megengine.functional as F

def arange_pixels(resolution=(128, 128), batch_size=1, image_range=(-1., 1.),
                  subsample_to=None, invert_y_axis=False):
    ''' Arranges pixels for given resolution in range image_range.

    The function returns the unscaled pixel locations as integers and the
    scaled float values.

    Args:
        resolution (tuple): image resolution
        batch_size (int): batch size
        image_range (tuple): range of output points (default [-1, 1])
        subsample_to (int): if integer and > 0, the points are randomly
            subsampled to this value
    '''
    h, w = resolution
    n_points = resolution[0] * resolution[1]

    # Arrange pixel location in scale resolution
    x_list = np.linspace(0., w - 1., w).reshape(1, w, 1)
    x_list = x_list.repeat(h, axis=0)
    y_list = np.linspace(0., h - 1., h).reshape(h, 1, 1)
    y_list = y_list.repeat(w, axis=1)
    xy_list = np.concatenate((x_list, y_list), 2)  # [H,W,2]
    pixel_locations = F.repeat(megengine.tensor(xy_list, dtype="float32").reshape((1, -1, 2)), repeats=batch_size, axis=0)
    pixel_scaled = F.copy(pixel_locations)

    # Shift and scale points to match image_range
    scale = (image_range[1] - image_range[0])
    loc = scale / 2
    pixel_scaled[:, :, 0] = scale * pixel_scaled[:, :, 0] / (w - 1) - loc
    pixel_scaled[:, :, 1] = scale * pixel_scaled[:, :, 1] / (h - 1) - loc

    # Subsample points if subsample_to is not None and > 0
    if (subsample_to is not None and subsample_to > 0 and subsample_to < n_points):
        raise NotImplementedError("do not implement")

    if invert_y_axis:
        assert(image_range == (-1, 1))
        pixel_scaled[..., -1] *= -1.
        pixel_locations[..., -1] = (h - 1) - pixel_locations[..., -1]

    return pixel_locations, pixel_scaled

def transform_to_world(pixels, depth, camera_mat, world_mat, scale_mat=None,
                       invert=True, use_absolute_depth=True):
    ''' Transforms pixel positions p with given depth value d to world coordinates.

    Args:
        pixels (tensor): pixel tensor of size B x N x 2
        depth (tensor): depth tensor of size B x N x 1
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: true)
    '''
    assert(pixels.shape[-1] == 2)

    if scale_mat is None:
        scale_mat = torch.eye(4).unsqueeze(0).repeat(
            camera_mat.shape[0], 1, 1).to(camera_mat.device)

    # Convert to pytorch
    pixels, is_numpy = to_pytorch(pixels, True)
    depth = to_pytorch(depth)
    camera_mat = to_pytorch(camera_mat)
    world_mat = to_pytorch(world_mat)
    scale_mat = to_pytorch(scale_mat)

    # Invert camera matrices
    if invert:
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)

    # Transform pixels to homogen coordinates
    pixels = pixels.permute(0, 2, 1) # b,2,x
    pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1) # b,4,x

    # Project pixels into camera space
    if use_absolute_depth:
        pixels[:, :2] = pixels[:, :2] * depth.permute(0, 2, 1).abs()
        pixels[:, 2:3] = pixels[:, 2:3] * depth.permute(0, 2, 1)
    else:
        pixels[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)

    # Transform pixels to world space
    p_world = scale_mat @ world_mat @ camera_mat @ pixels

    # Transform p_world back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1)

    if is_numpy:
        p_world = p_world.numpy()
    return p_world

def origin_to_world(n_points, camera_mat, world_mat, scale_mat=None,
                    invert=False):
    ''' Transforms origin (camera location) to world coordinates.

    Args:
        n_points (int): how often the transformed origin is repeated in the
            form (batch_size, n_points, 3)
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert the matrices (default: true)
    '''
    batch_size = camera_mat.shape[0]
    # Create origin in homogen coordinates
    p = torch.zeros(batch_size, 4, n_points)
    p[:, -1] = 1.

    if scale_mat is None:
        scale_mat = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)

    # Invert matrices
    if invert:
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)

    # Apply transformation
    p_world = scale_mat @ world_mat @ camera_mat @ p

    # Transform points back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1)
    return p_world

def image_points_to_world(image_points, camera_mat, world_mat, scale_mat=None,
                          invert=False, negative_depth=True):
    ''' Transforms points on image plane to world coordinates.

    In contrast to transform_to_world, no depth value is needed as points on
    the image plane have a fixed depth of 1.

    Args:
        image_points (tensor): image points tensor of size B x N x 2
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: true)
    '''
    batch_size, n_pts, dim = image_points.shape
    assert(dim == 2)
    d_image = F.ones((batch_size, n_pts, 1))

    if negative_depth:
        d_image *= -1.

    return transform_to_world(image_points, d_image, camera_mat, world_mat, scale_mat, invert=invert)

if __name__ == "__main__":
    pass
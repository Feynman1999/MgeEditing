import numpy as np
import megengine as mge
from scipy.spatial.transform import Rotation as Rot
import megengine.functional as F

def get_camera_mat(fov=49.13, invert=True):
    # fov = 2 * arctan( sensor / (2 * focal))
    # focal = (sensor / 2)  * 1 / (tan(0.5 * fov))
    # in our case, sensor = 2 as pixels are in [-1, 1]
    focal = 1. / np.tan(0.5 * fov * np.pi/180.)
    focal = focal.astype(np.float32)
    mat = mge.tensor([
        [focal, 0., 0., 0.],
        [0., focal, 0., 0.],
        [0., 0., 1, 0.],
        [0., 0., 0., 1.]
    ]).reshape(1, 4, 4)

    if invert:
        mat = F.matinv(mat)
    return mat

def get_random_pose(range_u, range_v, range_radius, batch_size=32, invert=False):
    loc = sample_on_sphere(range_u, range_v, size=(batch_size)) # 单位圆上batch个点 [32, 3]
    radius = range_radius[0] + (range_radius[1] - range_radius[0]) * mge.random.uniform(size=(batch_size, 1)) # 随机乘上距离
    loc = loc * radius
    R = look_at(loc.numpy())
    RT = F.repeat(F.eye(4).reshape(1, 4, 4), repeats=batch_size, axis=0) # [b, 4, 4] 
    RT[:, :3, :3] = R
    RT[:, :3, -1] = loc
    """
        1 0 0 0
        0 1 0 0
        0 0 1 0
        0 0 0 1
    """
    if invert:
        RT = F.matinv(RT)
    return RT

def to_sphere(u, v):
    """
        xy平面上一个角度 theta
        z轴一个角度 phi
        得到单位园上的一个点
    """
    theta = 2 * np.pi * u # [0, 2pi]
    phi = np.arccos(1 - 2 * v) # [0, pi]
    cx = np.sin(phi) * np.cos(theta)
    cy = np.sin(phi) * np.sin(theta)
    cz = np.cos(phi)
    return np.stack([cx, cy, cz], axis=-1)

def sample_on_sphere(range_u=(0, 1), range_v=(0, 1), size=(1,), to_mge=True):
    u = np.random.uniform(*range_u, size=size)
    v = np.random.uniform(*range_v, size=size)

    sample = to_sphere(u, v)

    if to_mge:
        sample = mge.tensor(sample, dtype="float32")

    return sample

def look_at(eye, at=np.array([0, 0, 0]), up=np.array([0, 0, 1]), eps=1e-5, to_mge=True):
    at = at.astype(float).reshape(1, 3) # [1, 3]
    up = up.astype(float).reshape(1, 3)
    eye = eye.reshape(-1, 3)
    up = up.repeat(eye.shape[0] // up.shape[0], axis=0) # [b, 3]
    eps = np.array([eps]).reshape(1, 1).repeat(up.shape[0], axis=0) # [b,1]

    z_axis = eye - at
    z_axis /= np.max(np.stack([np.linalg.norm(z_axis, axis=1, keepdims=True), eps]), axis = 0)

    x_axis = np.cross(up, z_axis)
    x_axis /= np.max(np.stack([np.linalg.norm(x_axis, axis=1, keepdims=True), eps]), axis = 0)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.max(np.stack([np.linalg.norm(y_axis, axis=1, keepdims=True), eps]), axis = 0)

    r_mat = np.concatenate(
        (x_axis.reshape(-1, 3, 1), y_axis.reshape(-1, 3, 1), z_axis.reshape(-1, 3, 1)) , axis=2)

    if to_mge:
        r_mat = mge.tensor(r_mat, dtype="float32")
    
    return r_mat


if __name__ == "__main__":
    # ans = look_at(np.array([[1., 1., 1.], [2., 2., 3.]]), to_mge=False)
    # print(ans)
    # print(np.linalg.det(ans))
    ans = get_random_pose(range_u=(0.5, 0.5), range_v=(0.5, 0.5), range_radius=(1, 1), batch_size=1)
    print(np.dot(ans.numpy(), ans))
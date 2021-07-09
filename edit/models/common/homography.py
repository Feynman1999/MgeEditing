import megengine as mge
import megengine.functional as F
import numpy as np

def DLT_solve(src_ps, off_sets):
    # src_p: shape=(N, 4, 2)
    # off_set: shape=(N, 4, 2)
    # can be used to compute mesh points (multi-H)
    N, h, w = src_ps.shape
    assert h == 4 and w == 2
    dst_p = src_ps + off_sets
    ones = F.ones((N, 4, 1))
    xy1 = F.concat([src_ps, ones], axis=2)
    zeros = F.zeros_like(xy1)
    xyu, xyd = F.concat([xy1, zeros], 2), F.concat([zeros, xy1], 2)
    M1 = F.concat([xyu, xyd], 2).reshape(N, -1, 6)
    M2 = F.matmul(
        dst_p.reshape(-1, 2, 1),
        src_ps.reshape(-1, 1, 2),
    ).reshape(N, -1, 2)
    A = F.concat([M1, -M2], 2)
    b = dst_p.reshape(N, -1, 1)
    Ainv = F.matinv(A)
    h8 = F.matmul(Ainv, b).reshape(N, 8)
    H = F.concat([h8, ones[:,0,:]], 1).reshape(N, 3, 3)
    return H

if __name__ == "__main__":
    B = 1
    h = 11
    w = 11
    block_h = 9
    block_w = 16
    H = (h-1) * block_h # train: 90
    W = (w-1) * block_w # train: 160
    x_list = np.arange(0., W - 1., block_w).reshape(1, w-1, 1) # train: 10个
    y_list = np.arange(0., H - 1., block_h).reshape(h-1, 1, 1) # train: 10个
    x_list = x_list.repeat(h-1, axis=0) # (10,10, 1)
    y_list = y_list.repeat(w-1, axis=1) # (10,10, 1)
    xy_list_up_left = np.concatenate((x_list, y_list), 2).reshape(-1, 2)  # [100,2]
    xy_list_up_right = np.concatenate((x_list+block_w-1, y_list), 2).reshape(-1, 2)  # [100,2]
    xy_list_down_left = np.concatenate((x_list, y_list + block_h-1), 2).reshape(-1, 2)  # [100,2]
    xy_list_down_right = np.concatenate((x_list + block_w - 1, y_list + block_h-1), 2).reshape(-1, 2)  # [100,2]
    # 按照顺序stack成[100,4,2]
    xy_list = np.stack([xy_list_up_left, xy_list_up_right, xy_list_down_left, xy_list_down_right], axis=1) # [100,4,2]
    # repeat batch times
    xy_list = xy_list.repeat(B, axis=0) # (x, 4, 2)
    print(xy_list[0:10])
    # src = [[0, 0], [1, 0], [0, 1], [1, 1]]
    # src = np.array(src).astype(np.float32)
    # dst = [[1, 0], [2, 0], [1, 1], [2, 1]]
    # dst = np.array(dst).astype(np.float32)

    # ###############

    # src = mge.tensor(src)
    # src = F.expand_dims(src, axis=0)
    # dst = mge.tensor(dst)
    # dst = F.expand_dims(dst, axis=0)

    # ###############
    
    # ans = DLT_solve(src, dst - src)
    # print(ans)

    # ###############
    # src = F.concat([src, F.ones((1, 4, 1))], axis=2).transpose(0,2,1) # [B, 3,4]
    # print(src.shape)
    # print(F.matmul(ans, src).transpose(0,2,1)[:, :, 0:2])
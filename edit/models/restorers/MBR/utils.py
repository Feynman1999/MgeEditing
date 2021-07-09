import megengine.functional as F
import megengine
from edit.utils import imwrite, tensor2img, flow_to_image
import numpy as np
import cv2 

def get_bilinear(image):
    B,T,C,h,w = image.shape
    image = image.reshape(-1, C,h,w)
    return F.nn.interpolate(image, scale_factor=4).reshape(B,T,C,4*h, 4*w)

viz_iter = 0
backwarp_tenGrid = {}

def get_grid_by_shape(shape):
    _, _, H, W = shape
    if str(shape) not in backwarp_tenGrid.keys():
        x_list = np.linspace(0., W - 1., W).reshape(1, 1, 1, W)
        x_list = x_list.repeat(H, axis=2)
        y_list = np.linspace(0., H - 1., H).reshape(1, 1, H, 1)
        y_list = y_list.repeat(W, axis=3)
        xy_list = np.concatenate((x_list, y_list), 1)  # [1,2,H,W]
        backwarp_tenGrid[str(shape)] = megengine.tensor(xy_list.astype(np.float32))
    return backwarp_tenGrid[str(shape)]

def cal_flow_mask(flow, flow_back, thredhold = 3):
    """
        flow: 0>1       [B, 2, H, W]
        flow_back: 1>0  [B, 2, H, W]
        return          [B, 1, H, W]
        new_grid = grid + warp
        sample_new_flow in flow_back
        
    """
    border_mode = "CONSTANT"
    scalar = 1000 # 出界的肯定不对
    thredhold = thredhold**2 # 欧氏距离
    grid = get_grid_by_shape(flow.shape)
    mapxy = grid + flow
    # sample in flow_back
    new_flow = F.nn.remap(inp = flow_back, map_xy=mapxy.transpose(0, 2, 3, 1), border_mode=border_mode, scalar = scalar)
    # [B, 2, H, W]
    back_grid = mapxy + new_flow
    mask = (back_grid[:, 0:1, ...] - grid[:, 0:1, ...])**2 + (back_grid[:, 1:2, ...] - grid[:, 1:2, ...])**2
    return (mask < thredhold).astype("float32")

def warp_image(now_img, ref_img, flow, mask = None):
    # now_img: [1, c, h, w]
    # ref_img: [1, c, h, w]
    # flow: [1, 2, h, w]
    # mask: [1, 1, h, w]
    border_mode = "CONSTANT"
    scalar = 0
    grid = get_grid_by_shape(flow.shape)
    mapxy = grid + flow
    warped_img = F.nn.remap(inp = ref_img, map_xy=mapxy.transpose(0, 2, 3, 1), border_mode=border_mode, scalar = scalar)
    if mask:
        return warped_img * mask
    else:
        return warped_img

def viz_flow_train_generator_batch(image, label, *, gm, netG, netloss):
    """
         save origin image and flow and mask
    """
    global viz_iter
    B,T,_,h,w = image.shape
    assert B == 1
    # cal all flows
    frames_1 = [] # [B, 3, H, W]
    frames_2 = [] # [B, 3, H, W]
    for i in range(1, T):
        frames_1.append(image[:, i, ...])
        frames_2.append(image[:, i-1, ...])
    flows_1_0 = netG.flownet(F.concat(frames_1, axis=0), F.concat(frames_2, axis=0)) # [(T-1)*B, h, w]
    frames_1 = []
    frames_2 = []
    for i in range(T-1):
        frames_1.append(image[:, i, ...])
        frames_2.append(image[:, i+1, ...])
    flows_0_1 = netG.flownet(F.concat(frames_1, axis=0), F.concat(frames_2, axis=0)) # [(T-1)*B, h, w]

    # take forward as example
    for i in range(T-1):
        warped_img = warp_image(image[0:1, i, ...], image[0:1, i+1, ...], flows_0_1[i*B:(i+1)*B, ...]) # [1, c, h, w]
        img_i = tensor2img(image[0, i])   # (h,w,3)
        img_i_plus_1 = tensor2img(image[0, i+1])
        warped_img = tensor2img(warped_img[0])
        flow = flows_0_1[i*B:(i+1)*B, ...]
        flow_back = flows_1_0[i*B:(i+1)*B, ...]
        flow_img = flow[0].transpose(1,2,0).numpy()
        flow_img = flow_to_image(flow_img)
        concat_imgs_line1 = [img_i, img_i_plus_1, flow_img]
        concat_imgs_line2 = [img_i, img_i_plus_1, warped_img]

        thredholds = [9, 7, 5, 3, 1]
        for thr in thredholds:
            mask = cal_flow_mask(flow, flow_back, thredhold = thr)
            mask = mask[0].transpose(1,2,0).numpy() # [H,W,1]
            flow_img_thr = flow_img * mask
            flow_img_thr = flow_img_thr.astype(np.uint8)
            flow_img_thr = cv2.putText(flow_img_thr, "thr= " + str(thr), (30, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            concat_imgs_line1.append(flow_img_thr)
            # add mask
            warp_img_thr = warped_img * mask
            warp_img_thr = warp_img_thr.astype(np.uint8)
            concat_imgs_line2.append(warp_img_thr)
        img_line1 = np.concatenate(concat_imgs_line1, axis=1)
        img_line2 = np.concatenate(concat_imgs_line2, axis=1)
        img = np.concatenate([img_line1, img_line2], axis = 0)
        imwrite(img, file_path="./workdirs/viz/iter_{}/{}_{}.png".format(viz_iter, i, i+1))

    viz_iter += 1
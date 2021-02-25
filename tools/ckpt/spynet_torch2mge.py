"""
    convert torch spynet params to mge file
    test result for spynet
"""
import torch
import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_path)
from edit.models.backbones.restorer_backbones.basicVSR import Spynet
import megengine as mge
from edit.utils import imread, flow_to_image, imwrite
import numpy as np
import megengine.functional as F

if __name__ == "__main__":
    mge.set_default_device('gpu6')
    """
        path = "./workdirs/spynet/network-sintel-final.pytorch"
        state_dict_torch = torch.load(path)
        state_dict_torch = {
            k.replace("module", "net"): v.numpy() for k,v in state_dict_torch.items()
        }
        net = Spynet(num_layers=6)
        state_dict_mge = net.state_dict()
        net.load_state_dict({
                k: v.reshape(state_dict_mge[k].shape)
                for k, v in state_dict_torch.items()
            }, strict=True)
        mge.save(net.state_dict(), "./workdirs/spynet/spynet-sintel-final.mge")
    """
    net = Spynet(num_layers=6)
    net.load_state_dict(mge.load("./workdirs/spynet/spynet-sintel-final.mge"), strict=True)
    net.eval()
    # read two images
    image1 = "./workdirs/spynet/2_64.png"
    image2 = "./workdirs/spynet/3_64.png"
    image1 = imread(image1, channel_order='rgb').astype(np.float32) / 255.
    image2 = imread(image2, channel_order='rgb').astype(np.float32) / 255.
    image1 = F.expand_dims(mge.tensor(image1), axis=0).transpose(0, 3, 1, 2)
    image2 = F.expand_dims(mge.tensor(image2), axis=0).transpose(0, 3, 1, 2)
    flow = net(image1, image2)
    flow = flow[0].transpose(1,2,0).numpy()
    # print(flow[0:10,0:10,0])
    print(np.max(flow), np.min(flow))
    flow = flow_to_image(flow)
    save_path = "./workdirs/spynet/flow_reds_64_6.png"
    imwrite(flow[:, :, [2,1,0]], save_path)

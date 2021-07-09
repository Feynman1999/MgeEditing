"""
1.读取pth文件，获得参数，转为numpy
2.进行格式调整，适合meg
3.保存为.mge
"""
import torch
import megengine
import numpy as np

torch_path = "./workdirs/gen_00025.pth"  
state_dict_torch = torch.load(torch_path)['netG']

mge_path = "./workdirs/sttn_official_fvi/20210522_140537/checkpoints/epoch_50/generator_module.mge"
state_dict_mge = megengine.load(mge_path)
for k,v in state_dict_mge.items():
    if "lastconv" in k:
        aim_k = k.replace("lastconv", "decoder.6")
    else:
        aim_k = k
    aim_v = state_dict_torch[aim_k].cpu().numpy()
    if len(aim_v.shape) == 1:
        # (64) to (1,64,1,1)
        aim_v =  np.expand_dims(aim_v, 0)
        aim_v =  np.expand_dims(aim_v, 2)
        aim_v =  np.expand_dims(aim_v, 3)
        print(aim_v.shape)
    state_dict_mge[k] = aim_v

megengine.save(state_dict_mge, "./workdirs/sttn.mge")
# state_dict_torch = {
#     k.replace("module", "net"): v.numpy() for k,v in state_dict_torch.items()
# }
# net = Spynet(num_layers=6)
# state_dict_mge = net.state_dict()
# net.load_state_dict({
#         k: v.reshape(state_dict_mge[k].shape)
#         for k, v in state_dict_torch.items()
#     }, strict=True)
# mge.save(net.state_dict(), "./workdirs/spynet/spynet-sintel-final.mge")
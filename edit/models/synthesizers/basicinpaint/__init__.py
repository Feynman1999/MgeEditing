import megengine.functional as F
from edit.utils import imwrite, tensor2img
from .onlyg import train_batch

def test_batch(frames, masks, netG):
    b, T, c, h, w = frames.shape
    assert h%8 + w%8 == 0
    masked_frame = (frames * (1 - masks))
    netG.train()
    now_hiddens = []
    hiddens = []
    for i in range(3):
        hiddens.append([]) # 存储前三层的所有hidden
    # get features from rgb
    masked_frame = F.concat([masked_frame, masks], axis=2)
    rgb = masked_frame.reshape(b*T, 4, h, w)
    rgb = netG.conv1(netG.conv0(rgb)).reshape(b, T, -1, h//2, w//2)
    for t in range(T):
        now_hiddens.append(rgb[:, t, ...])
    del rgb

    for layer_idx in range(3):
        """
            前三层没有shortcut, 结果会存放在hiddens中
        """ 
        #　存储当前的所有输入到hiddens中
        for item in now_hiddens:
            hiddens[layer_idx].append(item)

        if layer_idx % 2 == 0:
            hidden = now_hiddens[0] # [B,C,h,w]
            flow = None
            for i in range(T):
                now_hiddens[i] = netG.layers[layer_idx](now_hiddens[i], hidden, flow = flow)
                hidden = now_hiddens[i]
        else:
            hidden = now_hiddens[-1] # [B,C,h,w]
            flow = None
            for i in range(T-1, -1, -1):
                now_hiddens[i] = netG.layers[layer_idx](now_hiddens[i], hidden, flow = flow)
                hidden = now_hiddens[i]
        
        # 为下一层输入做准备
        for i in range(T):
            now_hiddens[i] = netG.layers[layer_idx].feature_output(now_hiddens[i])

    for layer_idx in range(3, 6):
        """
            后三层concat前面的shortcut
        """ 
        if layer_idx % 2 == 0:
            hidden = now_hiddens[0] # [B,C,h,w]
            flow = None
            for i in range(T):
                now_hidden = F.concat([now_hiddens[i], hiddens[5 - layer_idx][i]], axis=1)
                now_hiddens[i] = netG.layers[layer_idx](now_hidden, hidden, flow = flow)
                hidden = now_hiddens[i]
        else:
            hidden = now_hiddens[-1] # [B,C,h,w]
            flow = None
            for i in range(T-1, -1, -1):
                now_hidden = F.concat([now_hiddens[i], hiddens[5 - layer_idx][i]], axis=1)
                now_hiddens[i] = netG.layers[layer_idx](now_hidden, hidden, flow = flow)
                hidden = now_hiddens[i]
        
        # 为下一层输入做准备
        for i in range(T):
            now_hiddens[i] = netG.layers[layer_idx].feature_output(now_hiddens[i])

    for i in range(T):
        now_hiddens[i] = netG.do_upsample(now_hiddens[i])
    pred_img = F.stack(now_hiddens, axis = 1) # [B,T,3,H,W]
    pred_img = pred_img.reshape(b*T, c, h, w)
    frames = frames.reshape(b*T, c, h, w)
    masks = masks.reshape(b*T, 1, h, w)
    
    comp_img = frames*(1.-masks) + masks*pred_img
    return comp_img
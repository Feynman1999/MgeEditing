import megengine.functional as F
from edit.utils import imwrite, tensor2img

iter = 0

def train_batch(frames, masks, *, gm_G, gm_D, netG, netD, 
                optim_G, optim_D, pixel_loss, adv_loss, loss_weight):
    """
        先更新G再更新D
    """
    global iter
    iter = iter + 1
    b, T, c, h, w = frames.shape
    assert h%8 + w%8 == 0
    masked_frame = (frames * (1 - masks))
    for i in range(T):
        imwrite(tensor2img(masked_frame[0, i], min_max=(-1, 1)), file_path="./workdirs/{}_{}_mask.png".format(iter, i))
    
    gen_loss = 0
    dis_loss = 0
    netG.train()
    netD.train()
    with gm_G:
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

        # for i in range(T):
        #     imwrite(tensor2img(comp_img[i], min_max=(-1, 1)), file_path="./workdirs/{}_{}_res.png".format(iter, i))

        gen_vid_feat = netD(F.concat([comp_img, masks], axis=1))
        gan_loss = adv_loss(gen_vid_feat, True, False)
        gan_loss = gan_loss * loss_weight['adversarial_weight']
        gen_loss += gan_loss

        hole_loss = pixel_loss(pred_img*masks, frames*masks)
        hole_loss = hole_loss / masks.mean() * loss_weight['hole_weight']
        gen_loss += hole_loss 

        valid_loss = pixel_loss(pred_img*(1-masks), frames*(1-masks))
        valid_loss = valid_loss / (1-masks).mean() * loss_weight['valid_weight']
        gen_loss += valid_loss 

        optim_G.clear_grad()
        gm_G.backward(gen_loss)
        optim_G.step()

    # 优化D
    with gm_D:
        real_vid_feat = netD(F.concat([frames, masks], axis=1))
        fake_vid_feat = netD(F.concat([comp_img, masks], axis=1))
        dis_real_loss = adv_loss(real_vid_feat, True, True)
        dis_fake_loss = adv_loss(fake_vid_feat, False, True)
        # print(dis_real_loss, dis_fake_loss)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2
        optim_D.clear_grad()
        gm_D.backward(dis_loss)
        optim_D.step()
        
    return [hole_loss, valid_loss, gan_loss, dis_real_loss, dis_fake_loss]

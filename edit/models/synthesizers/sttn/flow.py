import megengine.functional as F

def train_batch(frames, masks, *, gm_G, gm_D, netG, netD, 
                optim_G, optim_D, pixel_loss, adv_loss, loss_weight):
    """
        先更新G再更新D
    """
    b, t, c, h, w = frames.shape
    masked_frame = (frames * (1 - masks))
    gen_loss = 0
    dis_loss = 0
    netG.train()
    netD.train()
    # 使用netG中的flownet求出光流
    flows = []
    for i in range(t-1):
        flows.append(netG.flownet(frames[:, i, :, :, :], frames[:, i+1, :, :, :]))
    flows = F.stack(flows, axis=1) # [b,t-1,2,h,w]
    flows = flows.reshape(b*(t-1), 2, h, w)
    with gm_G:
        pred_img = netG(masked_frame, masks)
        masks = masks[:, 0:t-1, :, :, :].reshape(b*(t-1), 1, h, w)
        comp_img = pred_img

        gen_vid_feat = netD(comp_img)
        gan_loss = adv_loss(gen_vid_feat, True, False)
        gan_loss = gan_loss * loss_weight['adversarial_weight']
        gen_loss += gan_loss

        hole_loss = pixel_loss(pred_img*masks, flows*masks)
        hole_loss = hole_loss / masks.mean() * loss_weight['hole_weight']
        gen_loss += hole_loss 

        valid_loss = pixel_loss(pred_img*(1-masks), flows*(1-masks))
        valid_loss = valid_loss / (1-masks).mean() * loss_weight['valid_weight']
        gen_loss += valid_loss 

        optim_G.clear_grad()
        gm_G.backward(gen_loss)
        optim_G.step()

    # 优化D
    with gm_D:
        real_vid_feat = netD(flows)
        fake_vid_feat = netD(comp_img)
        dis_real_loss = adv_loss(real_vid_feat, True, True)
        dis_fake_loss = adv_loss(fake_vid_feat, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2
        optim_D.clear_grad()
        gm_D.backward(dis_loss)
        optim_D.step()
        
    return [hole_loss, valid_loss, gan_loss, dis_real_loss, dis_fake_loss]


def test_batch(frames, masks, netG):
    b, t, c, h, w = frames.shape
    masked_frame = (frames * (1 - masks))
    netG.eval()
    pred_img = netG(masked_frame, masks)
    return pred_img

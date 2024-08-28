import cv2
import os
import math
import argparse
import numpy as np

import torch as th
import torch.nn.functional as F
import time
import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util
from skimage.metrics import structural_similarity as ssim
import skimage.io as io

# Workaround
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass


from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)  # noqa: E402

def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample


def main(conf: conf_mgt.Default_Conf):

    print("Start", conf['name'])

    device = dist_util.dev(conf.get('device'))

    # import ipdb;ipdb.set_trace()
    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = conf.show_progress
    cond_fn = None

    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None, 'y cann\'t be None'
        return model(x, t, y if conf.class_cond else None, gt=gt)

    print("sampling...")
    all_images = []
    eval_name = conf.get_default_eval_name()

    from my_data_utils import load_dataloader
    file_path = '../raw_data/cremi/sample_A_20160501.hdf'
    # file_path = '../data/fib25_xyz/2696_2952_1280_1536_2304_2560.tif'
    # file_path = '../data/cfm'
    # file_path = '../zz.png'
    batch_size = 2
    # tile_size = [32, 128]
    # gap_size = [32,128]
    # sr_ratio = 4
    # resize = True
    tile_size = [128, 128]
    gap_size = [128,128]
    sr_ratio = 4
    resize = False
    tile_size = [256, 256]
    gap_size = [256,256]
    sr_ratio = 4
    resize = False
    dl = load_dataloader(file_path, batch_size, tile_size, gap_size, sr_ratio, resize, mode='xy')

    psnr_resize=[]
    psnr_ddpm=[]
    ssim_resize=[]
    ssim_ddpm=[]
    for batch in iter(dl):
        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        model_kwargs = {}

        model_kwargs["gt"] = batch['GT']

        gt_keep_mask = batch.get('gt_keep_mask')
        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask

        batch_size = model_kwargs["gt"].shape[0]

        if conf.cond_y is not None:
            classes = th.ones(batch_size, dtype=th.long, device=device)
            model_kwargs["y"] = classes * conf.cond_y
        else:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(batch_size,), device=device
            )
            model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )

        import ipdb;ipdb.set_trace()
        result = sample_fn(
            model_fn,
            (batch_size, 3, conf.image_size, conf.image_size),
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=show_progress,
            return_all=True,
            conf=conf,
            model_instance=model
        )
        srs = toU8(result['sample'])
        # import ipdb;ipdb.set_trace()
        gts = toU8(result['gt'])

        lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *
                   th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))

        gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1))

        conf.eval_imswrite(
            srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks,
            img_names=batch['GT_name'], dset='eval', name=eval_name, verify_same=False)
        
        # # import ipdb;ipdb.set_trace()
        save_path = conf.data['eval'][eval_name]['paths']['srs'].replace('/inpainted','')
        os.makedirs(save_path+'/resize', exist_ok=True)
        resize_img = batch['resize_img'].cpu().numpy()

        os.makedirs(save_path+'/show', exist_ok=True)
        import ipdb;ipdb.set_trace()
        if True:
            mask = model_kwargs.get('gt_keep_mask').cpu().numpy()[0][0]
            for i in range(resize_img.shape[0]):
                name = batch['GT_name'][i].split('.')[0]
                os.makedirs(save_path+'/show/%s'%name, exist_ok=True)
                gt_ = gts[i][:,:,0]

                res_stack = np.zeros((30+3,mask.shape[0],mask.shape[1]), dtype=np.uint8)
                for timestamp in result['show']:
                    if timestamp>=29:
                        continue
                    res = result['show'][timestamp][i][0]
                    res = np.clip((res + 1)*127.5, 0, 255).astype(np.uint8)

                    mask_ratio = max(0, min((28-timestamp)/10,1.0)) 
                    res = gt_*mask + res*(1-mask)* mask_ratio
                    res_stack[28-timestamp] = res
                    cv2.imwrite(save_path+'/show/%s/%d.png'%(name, timestamp+1), res)
                res_stack[30:33] = res_stack[29]
                io.imsave(save_path+'/show/%s.tif'%(name), res_stack)
                import ipdb;ipdb.set_trace()
        else:
            mask = model_kwargs.get('gt_keep_mask').cpu().numpy()[0]
            for i in range(resize_img.shape[0]):
                name = batch['GT_name'][i].split('.')[0]
                os.makedirs(save_path+'/show/%s'%name, exist_ok=True)
                gt_ = gts[i][:,:].transpose(2,0,1)
                cv2.imwrite(save_path+'/show/%s/input.png'%(name), gt_.transpose(1,2,0)[80:176,80:176,:][::4,:,:])

                res_stack = np.zeros((30+3,mask.shape[0],mask.shape[1]), dtype=np.uint8)
                for timestamp in result['show']:
                    # if timestamp>=29:
                    #     continue
                    res = result['show'][timestamp][i]
                    res = np.clip((res + 1)*127.5, 0, 255).astype(np.uint8)

                    mask_ratio = max(0, min((28-timestamp)/10,1.0)) 
                    res = gt_*mask + res*(1-mask)* mask_ratio
                    res_stack[28-timestamp] = res[0]
                    cv2.imwrite(save_path+'/show/%s/%d.png'%(name, timestamp+1), res.transpose(1,2,0)[80:176,80:176,:])
                res_stack[30:33] = res_stack[29]
                io.imsave(save_path+'/show/%s.tif'%(name), res_stack)

                import ipdb;ipdb.set_trace()
                if True:
                    mask =  model_kwargs.get('gt_keep_mask').cpu().numpy()[0][:,80:176,80:176]
                    t_k = result['show'][1][0][:,80:176,80:176]
                    t_k = np.clip((t_k + 1)*127.5, 0, 255).astype(np.uint8)
                    t_k_mask = (t_k*mask)
                    t_k_mask2 = (t_k*(1-mask))
                    # t_k_mask = (t_k*(1-mask))
                    

                    t_k_1 = result['show'][3][0][:,80:176,80:176]
                    t_k_1 = np.clip((t_k_1 + 1)*127.5, 0, 255).astype(np.uint8)
                    t_k_1_mask = (t_k_1*(1-mask))

                    gt = gts[0][:,:].transpose(2,0,1)[:,80:176,80:176]
                    gt_mask = (gt*mask)
                    # gt_noise = (t_k*mask)[:,80:176,80:176]

                    cv2.imwrite('1.png', t_k.transpose(1,2,0))
                    cv2.imwrite('2.png', t_k_mask.transpose(1,2,0))
                    cv2.imwrite('3.png', gt.transpose(1,2,0))
                    cv2.imwrite('4.png', gt_mask.transpose(1,2,0))
                    cv2.imwrite('5.png', t_k_1.transpose(1,2,0))
                    cv2.imwrite('6.png', t_k_mask2.transpose(1,2,0))
                    cv2.imwrite('7.png', mask.transpose(1,2,0))



        if '.tif' in batch['GT_name'][0]:
            resize_img = (resize_img/256.0).astype(np.uint8)
        for i in range(resize_img.shape[0]):
            name = batch['GT_name'][i]
            if '.tif' in name:
                name = name.split('.tif')[0]+'.png'
            cv2.imwrite(save_path+'/resize/%s'%(name), resize_img[i])
        
        import ipdb;ipdb.set_trace()
        
    #     # psnr(gts, srs)
    #     if True:
    #         img1 = gts
    #         img2 = resize_img
    #         mse = np.mean( (img1 - img2) ** 2 )
    #         PIXEL_MAX = 255.0
    #         psnr1 = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    #         psnr_resize.append(psnr1)
    #         ssim1 = ssim(img1[0][:,:,0], img2[0][:,:,0])
    #         ssim_resize.append(ssim1)
    #         img1 = gts
    #         img2 = srs
    #         mse = np.mean( (img1 - img2) ** 2 )
    #         PIXEL_MAX = 255.0
    #         psnr2 = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    #         psnr_ddpm.append(psnr2)
    #         ssim2 = ssim(img1[0][:,:,0], img2[0][:,:,0])
    #         ssim_ddpm.append(ssim2)

    #         print("PSNR: %4f, %4f" % (psnr1, psnr2))
    #         print("SSIM: %4f, %4f" % (ssim1, ssim2))

    # # import ipdb;ipdb.set_trace()
    # print('PSNR-->  cubic:%5f  ddpm:%5f'%(np.mean(np.array(psnr_resize)), np.mean(np.array(psnr_ddpm))))
    # print('SSIM-->  cubic:%5f  ddpm:%5f'%(np.mean(np.array(ssim_resize)), np.mean(np.array(ssim_ddpm))))
    # # print("Sampling completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get('conf_path')))
    main(conf_arg)

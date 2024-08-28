import random
import os
import glob
import h5py  

import cv2
import math
import numpy as np
import skimage.io as io
import torch
from torch.utils.data import DataLoader, Dataset

import pytorch_ssim
from torch.autograd import Variable
from multiprocessing import Pool

def func(data):
    [path, img] = data
    cv2.imwrite(path, img)

	

def generate_tiles(file_path, tile_size, gap_size, save_folder='work_dirs/tmp', mode='yz'):
    tile_t, tile_h = tile_size
    gap_t, gap_h = gap_size

    os.makedirs(save_folder, exist_ok=True)
    if '.png' in file_path:
        data = cv2.imread(file_path)
        data = data[:,:,0]
        H,W = data.shape
        mode='2d'
    else:
        if 'hdf' in file_path:
            with h5py.File(file_path,'r') as f:        
                data = np.copy(f['volumes']['raw'])
        else:
            data = io.imread(file_path)
        T,H,W = data.shape
        if T<tile_t:
            data_ = np.zeros((tile_t,H,W), dtype=data.dtype)
            data_[:T,:,:] = data
            data = data_
            T = tile_t
        

    # for i in range(data.shape[0]):
    #     cv2.imwrite('work_dirs/slices/%d.png'%(i), data[i])
    # import ipdb;ipdb.set_trace()
    data_list = []
    save_paths = []
    if mode=='yz' or mode=='xz':
        for w in range(W):
            for h in range(H//gap_h):
                for t in range(T//gap_t):
                    t0, h0 = t*gap_t, h*gap_h
                    t1, h1 = t0+tile_t, h0+tile_h
                    if t1>T or h1>H:
                        continue
                    save_path = '%s/%d_%d_%d_%d_%d_%d.png'%(save_folder,t0,t1,w,w,h0,h1)
                    save_paths.append(save_path)
                    data_list.append([save_path, data[t0:t1, h0:h1, w]])
                    if len(data_list)>=8:
                        break
                if len(data_list)>=8:
                    break
            if len(data_list)>=8:
                break
    elif mode=='xy':
        for t in range(T):
            for w in range(W//gap_h):
                for h in range(H//gap_h):
                
                    w0, h0 = w*gap_h, h*gap_h
                    w1, h1 = w0+tile_h, h0+tile_h
                    if w1>W or h1>H:
                        continue
                    save_path = '%s/%d_%d_%d_%d_%d_%d.png'%(save_folder,t,t,w0,w1,h0,h1)
                    save_paths.append(save_path)
                    data_list.append([save_path, data[t, h0:h1, w0:w1]])
                    # cv2.imwrite(, )
        # break
                    if len(data_list)>=8:
                        break
                if len(data_list)>=8:
                    break
            if len(data_list)>=8:
                break
    elif mode=='2d':
        for w in range(W//gap_h):
            for h in range(H//gap_h):
                w0, h0 = w*gap_h, h*gap_h
                w1, h1 = w0+tile_h, h0+tile_h
                if w1>W or h1>H:
                    continue
                save_path = '%s/%d_%d_%d_%d.png'%(save_folder,w0,w1,h0,h1)
                save_paths.append(save_path)
                data_list.append([save_path, data[h0:h1, w0:w1]])
    else:
        raise ValueError('mode:%s not surpported'%mode)

    
    with Pool(16) as p:
        p.imap(func, data_list)
        p.close()
        p.join()
        
    return save_paths



def load_dataloader(
        file_path,
        batch_size,
        tile_size,
        gap_size,
        sr_ratio,
        resize=False,
        mode='yz',
    ):
    # import ipdb;ipdb.set_trace()
    if os.path.isfile(file_path):
        save_paths = generate_tiles(file_path, tile_size, gap_size, mode=mode)
    else:
        save_paths = []
        for file_type in ['*.png', '*.tif']:
            save_paths += glob.glob(os.path.join(file_path, file_type))
    # save_paths = ['/home/notebook/data/personal/80383614/panmingjie/sr3d/RePaint/log/cremi2fib_yz/inpainted/AVG.png']
    print(len(save_paths))
    if len(save_paths)>16:
        save_paths=save_paths[:16]
    dataset = StackDataset(save_paths, sr_ratio, resize)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
    )



class StackDataset(Dataset):
    def __init__(self, files, sr_ratio, resize):
        super().__init__()
        self.files = files
        self.resize = resize
        self.sr_ratio=sr_ratio


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # import ipdb;ipdb.set_trace()
        path = self.files[idx]
        # import ipdb;ipdb.set_trace()
        if '.tif' in path:
            img = io.imread(path)
            h,w = img.shape
            normalize_value = 65535/2.0
        else:
            img = cv2.imread(path)
            h,w,c = img.shape
            normalize_value = 255/2.0

        if '.tif' in path:
            img = img.reshape(img.shape[0], img.shape[1], 1)


        if self.resize:
            img = cv2.resize(img, (w, h*self.sr_ratio))
            img_resize = np.copy(img)
        else:
            img_resize = cv2.resize(img[::self.sr_ratio,:,:], (w, h))

        img = img.astype(np.float32) / normalize_value - 1
        # import ipdb;ipdb.set_trace()
        img_resize = img_resize.astype(np.float32)# / normalize_value - 1
        
        if len(img_resize.shape)==2:
            img_resize = img_resize.reshape(img_resize.shape[0], img_resize.shape[1], 1)


        mask = np.zeros_like(img)
        mask[::self.sr_ratio, :, :] = 1.0

        name = os.path.basename(path)
        return {
            'GT': np.transpose(img, [2, 0, 1]),
            'GT_name': name,
            'gt_keep_mask': np.transpose(mask, [2, 0, 1]),
            'resize_img': img_resize,
        }




def cal_psnr(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
 
 
def cal_ssim(img1,img2):
    img1 = torch.from_numpy(img1).permute(0,3,1,2).float()/255.0
    img2 = torch.from_numpy(img2).permute(0,3,1,2).float()/255.0
    img1 = Variable( img1,  requires_grad=False)    # torch.Size([256, 256, 3])
    img2 = Variable( img2, requires_grad = False)
    ssim_value = pytorch_ssim.ssim(img1, img2).item()
    return ssim_value


if __name__=='__main__':

    file_path = '../raw_data/sample_A_20160501.hdf'
    batch_size = 8
    # tile_size = [32, 128]
    # gap_size = [32,128]
    # sr_ratio = 4
    # resize = True
    tile_size = [128, 128]
    gap_size = [128,128]
    sr_ratio = 2
    resize = False
    dl = load_dataloader(file_path, batch_size, tile_size, gap_size, sr_ratio, resize)
    for batch in iter(dl):
        print(batch.keys())
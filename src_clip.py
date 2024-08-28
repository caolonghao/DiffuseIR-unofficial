from skimage import io
import numpy as np
import cv2
src = io.imread('z3d_input8.tif')

# src = src[::2,:,:]

# mask = io.imread('data/2pm/mask/1.png')
print(src.shape)

up_ratio = 4

for i in range(src.shape[-1]):
    img = src[:,:,i]

    img = cv2.resize(img, (img.shape[1], img.shape[0]*2))
    
    res = np.zeros((img.shape[0]*up_ratio, img.shape[1]), dtype=img.dtype)
    res[::up_ratio] = img
    print(img.shape, res.shape)
    # if down_and_up:
    #     img = img[::2]
    # print(img.shape)
    io.imsave('data/2pm/gt/'+str(i)+'.png', res)
    # io.imsave('data/2pm/mask/'+str(i)+'.png', mask)

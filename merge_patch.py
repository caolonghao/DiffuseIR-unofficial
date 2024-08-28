import cv2
import glob,os
import numpy as np
import skimage.io as io

img_paths = glob.glob('log/test_inet256_thin/inpainted/*.png')
img_paths = ['tmp/'+str(idx)+'.png' for idx in range(300)]

img_list = []
for img_path in img_paths:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_list.append(img)
# img_paths.sort()
print(img_paths)
img_merge = np.stack(img_list, axis=2)
# import ipdb;ipdb.set_trace()
print(img_merge.shape)
io.imsave('solved2.tif', img_merge)

import glob
import cv2
import os
import numpy as np

files = glob.glob('log/test_inet256_thin/resize/*.png')

max_z, max_y, max_x = 0, 0, 0
for file in files:
    name = os.path.basename(file).split('.')[0]
    z0,z1,y0,y1,x0,x1 = map(int, name.split('_'))
    max_z = max(max_z, z1)
    max_y = max(max_y, y1)
    max_x = max(max_x, x1)
print(max_z, max_y, max_x)

output = np.zeros((max_z*4, max_x), dtype=np.uint8)

for file in files:
    name = os.path.basename(file).split('.')[0]
    z0,z1,y0,y1,x0,x1 = map(int, name.split('_'))

    img = cv2.imread(file)[:,:,0]
    # img = cv2.resize(img, (img.shape[1], img.shape[0]*4))
    output[z0*4:z1*4,x0:x1] = img
cv2.imwrite('zzz_resize.png', output)
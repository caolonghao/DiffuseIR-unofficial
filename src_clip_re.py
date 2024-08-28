from skimage import io
import glob, os

src = io.imread('src_256.tif')

src = src[::2,:,:]

mask = io.imread('data/2pm/mask_total/1.png')
print(src.shape)

src_list = glob.glob('data/2pm/gt_total/*.png')
src_list = [os.path.basename(path) for path in src_list]
tgt_list = glob.glob('log/test_inet256_thin/inpainted/*.png')
tgt_list = [os.path.basename(path) for path in tgt_list]


idx = 0
for i in range(len(src_list)):
    img_name = src_list[i]
    if img_name in tgt_list:
        continue
    print(img_name)
    idx += 1
    img = io.imread('data/2pm/gt_total/'+img_name)
    io.imsave('data/2pm/gt/'+img_name, img)
    io.imsave('data/2pm/mask/'+img_name, mask)

print(idx)
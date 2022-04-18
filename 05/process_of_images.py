# Optical flow images get resized and padded 

import util
from PIL import Image
import tv_hi_data as thd

size = (164, 164)

print('-- Saving images')
for i in range(len(thd.set_1)):
    for j in range(1,17):
        file_name = thd.set_1[i] + '_' + str(j) + '.jpg'
        img = Image.open('tv-hi_data/optical_flow/'+ file_name)
        img = util.resize_and_pad(img, size)
        img.save('tv-hi/optical-flow-proc/'+ file_name)
        img.close()

for i in range(len(thd.set_2)):
    for j in range(1,17):
        file_name = thd.set_2[i] + '_' + str(j) + '.jpg'
        img = Image.open('tv-hi_data/optical_flow/'+ file_name)
        img = util.resize_and_pad(img, size)
        img.save('tv-hi_data/optical_flow_proc/'+ file_name)
        img.close()
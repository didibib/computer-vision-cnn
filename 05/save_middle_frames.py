import tv_hi_data
import cv2
import util
from PIL import Image

def grey_pixels(img):
    grey = 0
    h = img.shape[0]
    w = img.shape[1]
    for i in range(w):
        for j in range(h):
            bgr = img[j,i]
            rng = range(129,132)
            if bgr[2] in rng and bgr[1] in rng and bgr[0] in rng:
                grey += 1
    return grey

def save_middle_frame(orig, dest):
    video = cv2.VideoCapture(orig)
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    video.set(cv2.CAP_PROP_POS_FRAMES, frames/2-1)
    _, img = video.read()
    
    width = img.shape[0]
    height = img.shape[1]
    counter = 0
    while(grey_pixels(img) > width * height * 0.75 and counter < 5):
        _, img = video.read()
        counter += 1
    cv2.imwrite(dest, img)

for item in tv_hi_data.set_1:
    save_middle_frame('tv-hi_data/videos/' + item + '.avi', 'tv-hi_data/middle_frames/' + item + '.jpg')

for item in tv_hi_data.set_2:
    save_middle_frame('tv-hi_data/videos/' + item + '.avi', 'tv-hi_data/middle_frames/' + item + '.jpg')

size = (164, 164)
for item in tv_hi_data.set_1:
    loc = 'tv-hi_data/middle_frames/'+ item + '.jpg'
    img = util.resize_and_pad(Image.open(loc), size)
    img.save(loc)

for item in tv_hi_data.set_2:
    loc = 'tv-hi_data/middle_frames/'+ item + '.jpg'
    img = util.resize_and_pad(Image.open(loc), size)
    img.save(loc)
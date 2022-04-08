from tensorflow import keras
from tensorflow.image import resize_with_pad
from keras.preprocessing.image import load_img
from keras.preprocessing.image import smart_resize
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv
from PIL import Image, ImageOps

# Stanford 40 data set
with open('st-40/testtrain/train.txt', 'r') as f:
    train_files = list(map(str.strip, f.readlines()))
    train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]
    print(f'Train files ({len(train_files)}):\n\t{train_files}')
    print(f'Train labels ({len(train_labels)}):\n\t{train_labels}\n')

with open('st-40/testtrain/test.txt', 'r') as f:
    test_files = list(map(str.strip, f.readlines()))
    test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]
    print(f'Test files ({len(test_files)}):\n\t{test_files}')
    print(f'Test labels ({len(test_labels)}):\n\t{test_labels}\n')
    
action_categories = sorted(list(set(['_'.join(name.split('_')[:-1]) for name in train_files])))
print(f'Action categories ({len(action_categories)}):\n{action_categories}')

# img = Image.open('st-40/images/test2.jpeg')
# img2 = ImageOps.fit(img, (224, 244), method = 0,
#                    bleed = 0.0, centering =(0.5, 0.5))

# plt.imshow(img2)
# plt.show()

        

# plt.imshow(img)
# plt.show()


# resized = resize_with_pad(img, 224, 224)
# plt.imshow(resized)
# plt.show()

def processImage(img, size):
    (dimX, dimY) = img.size
    maxDim = max(dimX, dimY)
    padded = ImageOps.pad(img, (maxDim, maxDim), color=(0,0,0))
    return padded.resize(size)

size = (224, 224)
l = lambda loc: processImage(Image.open('st-40/images/'+ loc), size)
train_images = list(map(l, train_files))
test_images = list(map(l, test_files))

plt.imshow(train_images[0])
plt.show()

# TV-Human interaction data set
# set_1_indices = [[2,14,15,16,18,19,20,21,24,25,26,27,28,32,40,41,42,43,44,45,46,47,48,49,50],
#                  [1,6,7,8,9,10,11,12,13,23,24,25,27,28,29,30,31,32,33,34,35,44,45,47,48],
#                  [2,3,4,11,12,15,16,17,18,20,21,27,29,30,31,32,33,34,35,36,42,44,46,49,50],
#                  [1,7,8,9,10,11,12,13,14,16,17,18,22,23,24,26,29,31,35,36,38,39,40,41,42]]
# set_2_indices = [[1,3,4,5,6,7,8,9,10,11,12,13,17,22,23,29,30,31,33,34,35,36,37,38,39],
#                  [2,3,4,5,14,15,16,17,18,19,20,21,22,26,36,37,38,39,40,41,42,43,46,49,50],
#                  [1,5,6,7,8,9,10,13,14,19,22,23,24,25,26,28,37,38,39,40,41,43,45,47,48],
#                  [2,3,4,5,6,15,19,20,21,25,27,28,30,32,33,34,37,43,44,45,46,47,48,49,50]]
# classes = ['handShake', 'highFive', 'hug', 'kiss']  # we ignore the negative class

# # test set
# set_1 = [f'tv-hi/{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_1_indices[c]]
# set_1_label = [f'{classes[c]}' for c in range(len(classes)) for i in set_1_indices[c]]
# print(f'Set 1 to be used for test ({len(set_1)}):\n\t{set_1}')
# print(f'Set 1 labels ({len(set_1_label)}):\n\t{set_1_label}\n')

# # training set
# set_2 = [f'tv-hi/{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_2_indices[c]]
# set_2_label = [f'{classes[c]}' for c in range(len(classes)) for i in set_2_indices[c]]
# print(f'Set 2 to be used for train and validation ({len(set_2)}):\n\t{set_2}')
# print(f'Set 2 labels ({len(set_2_label)}):\n\t{set_2_label}')
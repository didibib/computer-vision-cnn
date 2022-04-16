from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import cv2 as cv
from model import Model
import efficient_net as en
from PIL import Image

print('-- Reading ST-40 actions')
with open('st-40/actions.txt', 'r') as f:
    class_names = list(map(str.strip, f.readlines()))

print('-- Reading ST-40 processed images')
with open('st-40/train.txt', 'r') as f:
    train_files = list(map(str.strip, f.readlines()))    
train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]
train_images = [] #list(map(lambda loc: Image.open('st-40/proc-images/'+ loc), train_files))
for loc in train_files:
    img = load_img('st-40/proc-images/' + loc)
    train_images.append(img_to_array(img))

with open('st-40/test.txt', 'r') as f:
    test_files = list(map(str.strip, f.readlines()))
test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]
test_images = [] #list(map(lambda loc: Image.open('st-40/proc-images/'+ loc), test_files))
for loc in test_files:
    img = load_img('st-40/proc-images/' + loc)
    test_images.append(img_to_array(img))
    # file = Image.open('st-40/proc-images/' + loc)
    # test_images.append(img_to_array(file.copy()))
    # file.close()

print(train_images.shape())

print('-- Starting ST-40 model')
st_40_eff_net = Model(en.efficient_net(), class_names, 'st-40-effnet')
st_40_eff_net.run(train_images, train_labels, test_images, test_labels)



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
# Train on the Stanford 40 - Frames

import util
from model import Model
from p5_model import p5
from efficient_net import b0

print('-- Reading ST40 actions')
class_names = util.get_class_names('st-40/actions.txt')
dictionary = util.create_dict_index(class_names)

print('-- Reading ST40 processed train images')
with open('st-40/train.txt', 'r') as f:
    train_files = list(map(str.strip, f.readlines()))
train_labels_str = ['_'.join(name.split('_')[:-1]) for name in train_files]
train_labels = util.convert_to_index(train_labels_str, dictionary)
train_images = util.get_images('st-40/proc-images/', train_files)

print('-- Reading ST40 processed test images')
with open('st-40/test.txt', 'r') as f:
    test_files = list(map(str.strip, f.readlines()))
test_labels_str = ['_'.join(name.split('_')[:-1]) for name in test_files]
test_labels = util.convert_to_index(test_labels_str, dictionary)
test_images = util.get_images('st-40/proc-images/', test_files)

sh_test_lbl, sh_test_img = util.unison_shuffled_copies(test_labels, test_images)
sh_train_lbl, sh_train_img = util.unison_shuffled_copies(train_labels, train_images)

print('-- Starting ST40 model')
p5net = Model(b0(), class_names, 'st40_frames')
p5net.run(sh_train_img, sh_train_lbl, sh_test_img, sh_test_lbl)
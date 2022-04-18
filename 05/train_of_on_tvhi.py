import tv_hi_data as thd
import numpy as np
import tensorflow as tf
import util
from models.of_model import of
from keras.preprocessing.image import load_img
from models.model import Model
from keras import backend as K

# memory memes
K.clear_session()

def get_stacked_images(set):
    stacked = []
    for name in set:
        stack = [] #np.zeros((164,164,32))
        for i in range(0,16):
            path = 'tv-hi_data/optical_flow_proc/' + name + '_' + str(i+1) + '.jpg'
            img = load_img(path)
            img = tf.cast(img, tf.float32)
            img /= 255
            img = np.array(img)
            hv = np.zeros((img.shape[0], img.shape[1], 2))
            # ignore the saturation channel
            hv[:,:,0] = img[:,:,0]
            hv[:,:,1] = img[:,:,2]
            stack.append(hv)  
        # place temporal information on the third channel 
        # current shape [tmp, x, y, channel]
        # we want       [x, y, tmp, channel]
        stack = np.array(stack)
        stack = np.moveaxis(stack, 0, 2)#np.transpose(stack, (2, 0, 1, 3))          
        stacked.append(np.array(stack))
    return np.array(stacked)


print('-- Get stack of images')
train_images = get_stacked_images(thd.train_set)
test_images = get_stacked_images(thd.test_set)

sh_train_lbl, sh_train_img = util.unison_shuffled_copies(thd.training_labels, train_images)
sh_test_lbl, sh_test_img = util.unison_shuffled_copies(thd.test_labels, test_images)

print('-- Train of model')
model = Model(of, thd.classes, 'tvhi_of')
K.set_value(model._model.optimizer.lr, 0.0002)

model.run(sh_train_img, sh_train_lbl, sh_test_img, sh_test_lbl)
print(model._model.summary())
import util
import tv_hi_data as thd
from tensorflow import keras
from keras import backend as K
from models.model import Model

st40_frames = keras.models.load_model('models/st40_frames_10')

st40_frames.layers.pop() # Remove the output layer
st40_frames.layers.append(keras.layers.Dense(4))

# Set learning rate to 1/10th of original
K.set_value(st40_frames.optimizer.lr, st40_frames.optimizer.lr / 10)

training_images = util.get_images('tv-hi_data/middle_frames/', thd.set_1, '.jpg')
test_images = util.get_images('tv-hi_data/middle_frames/', thd.set_2, '.jpg')

sh_test_lbl, sh_test_img = util.unison_shuffled_copies(thd.test_labels, test_images)
sh_train_lbl, sh_train_img = util.unison_shuffled_copies(thd.training_labels, training_images)

model = Model(st40_frames, thd.classes, 'tvhi_frames')
model.run(sh_train_img, sh_train_lbl, sh_test_img, sh_test_lbl)

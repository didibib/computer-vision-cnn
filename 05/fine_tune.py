import util
import tv_hi_data as thd
from tensorflow import keras
from keras import backend as K
from model import Model

p5net = keras.models.load_model('models/p5net')

p5net.layers.pop() # Remove the output layer
p5net.layers.append(keras.layers.Dense(4))

# Set learning rate to 1/10th of original
K.set_value(p5net.optimizer.lr, p5net.optimizer.lr / 10)

class_dict = util.create_dict_index(thd.classes)
training_labels = util.convert_to_index(thd.set_2_label, class_dict)
test_labels = util.convert_to_index(thd.set_1_label, class_dict)

training_images = util.get_images('tv-hi/middle-frames/', thd.set_1, '.jpg')
test_images = util.get_images('tv-hi/middle-frames/', thd.set_2, '.jpg')

sh_test_lbl, sh_test_img = util.unison_shuffled_copies(test_labels, test_images)
sh_train_lbl, sh_train_img = util.unison_shuffled_copies(training_labels, training_images)

model = Model(p5net, thd.classes, 'p5net')
model.run(sh_train_img, sh_train_lbl, sh_test_img, sh_test_lbl)

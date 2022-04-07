from tensorflow import keras

import model
import variants as v

from keras.datasets import fashion_mnist
from model import Model

# Import training and validation sets
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

# Creating and training model
base_model = Model(model.base, class_names, 'base')
base_model.run(train_images, train_labels, test_images, test_labels)

variant1_model = Model(v.variant1, class_names, 'variant1_dense')
variant1_model.run(train_images, train_labels, test_images, test_labels)

variant2_model = Model(v.variant2, class_names, 'variant2_kernel')
variant2_model.run(train_images, train_labels, test_images, test_labels)

variant3_model = Model(v.variant3, class_names, 'variant3_lrelu')
variant3_model.run(train_images, train_labels, test_images, test_labels)

variant4_model = Model(v.variant4, class_names, 'variant4_batch_norm')
variant4_model.run(train_images, train_labels, test_images, test_labels)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import ImageOps
from keras.preprocessing.image import load_img
import numpy

def plot_image(i, predictions_array, true_label, img, class_names):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100*np.max(predictions_array),
                                    class_names[true_label]),
                                    color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def get_class_names(path):
    with open(path, 'r') as f:
        class_names = list(dict.fromkeys(list(map(str.strip, f.readlines()))))
    return class_names

def create_dict_index(keys):
    indices = list(range(len(keys)))
    zipKeyInd = zip(keys, indices)
    return dict(zipKeyInd)

def convert_to_index(labels, dictionary):
    indices = []
    for label in labels:
        index = dictionary[label]
        indices.append(index)
    return np.array(indices)

def get_images(directory, locations, type = ''):
    images = []
    for loc in locations:
        img = load_img(directory + loc + type)
        img = tf.cast(img, tf.float32)
        images.append(np.array(img))
    return np.array(images)

def resize_and_pad(img, size):
    (dimX, dimY) = img.size
    maxDim = max(dimX, dimY)
    padded = ImageOps.pad(img, (maxDim, maxDim), color=0)
    return padded.resize(size)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]